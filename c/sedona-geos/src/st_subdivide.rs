// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

// Constants following PostGIS implementation
const MAX_DEPTH: i32 = 50;
const FP_TOLERANCE: f64 = 1e-12; // For degenerate geometry handling
const MIN_MAX_VERTICES: i32 = 5;
const DEFAULT_MAX_VERTICES: i32 = 256; // PostGIS default
const DEFAULT_GRID_SIZE: f64 = -1.0; // No grid snapping by default (lwgeom_dump.c:343)

/// ST_Subdivide() implementation using the geos crate
/// Follows PostGIS implementation exactly (lwgeom.c:2430-2645)
pub fn st_subdivide_impl() -> ScalarKernelRef {
    Arc::new(STSubdivide {})
}

#[derive(Debug)]
struct STSubdivide {}

impl SedonaScalarKernel for STSubdivide {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_batch_impl(arg_types, args, DEFAULT_MAX_VERTICES, DEFAULT_GRID_SIZE)
    }
}

pub fn st_subdivide_with_max_vertices_impl() -> ScalarKernelRef {
    Arc::new(STSubdivideWithMaxVertices {})
}

#[derive(Debug)]
struct STSubdivideWithMaxVertices {}

impl SedonaScalarKernel for STSubdivideWithMaxVertices {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_integer()],
            WKB_GEOMETRY,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let max_vertices = extract_scalar_i32(args.get(1))?.unwrap_or(DEFAULT_MAX_VERTICES);
        invoke_batch_impl(arg_types, args, max_vertices, DEFAULT_GRID_SIZE)
    }
}

pub fn st_subdivide_with_grid_size_impl() -> ScalarKernelRef {
    Arc::new(STSubdivideWithGridSize {})
}

#[derive(Debug)]
struct STSubdivideWithGridSize {}

impl SedonaScalarKernel for STSubdivideWithGridSize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_numeric(),
            ],
            WKB_GEOMETRY,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let max_vertices = extract_scalar_i32(args.get(1))?.unwrap_or(DEFAULT_MAX_VERTICES);
        let grid_size = extract_scalar_f64(args.get(2))?.unwrap_or(DEFAULT_GRID_SIZE);
        invoke_batch_impl(arg_types, args, max_vertices, grid_size)
    }
}

fn invoke_batch_impl(
    arg_types: &[SedonaType],
    args: &[ColumnarValue],
    max_vertices: i32,
    grid_size: f64,
) -> Result<ColumnarValue> {

    // Validate: must be >= 5 (PostGIS line 2630-2634)
    if max_vertices < MIN_MAX_VERTICES {
        return Err(DataFusionError::Execution(format!(
            "max_vertices must be >= {}",
            MIN_MAX_VERTICES
        )));
    }

    let executor = GeosExecutor::new(arg_types, args);
    let mut builder = BinaryBuilder::with_capacity(
        executor.num_iterations(),
        WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
    );

    executor.execute_wkb_void(|wkb| {
        match wkb {
            Some(geom) => {
                // Get geometry dimension (PostGIS line 2636)
                // This is the topological dimension: 0=point, 1=line, 2=polygon
                let dimension = geom
                    .get_num_dimensions()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;

                // Subdivide recursively, accumulating results
                let mut results = Vec::new();
                subdivide_recursive(
                    &geom,
                    dimension as i32,
                    max_vertices,
                    0, // initial depth (PostGIS startdepth = 0)
                    &mut results,
                    grid_size,
                )?;

                // Create GeometryCollection from results (PostGIS line 2625-2637)
                let collection = if results.is_empty() {
                    geos::Geometry::create_empty_collection(geos::GeometryTypes::GeometryCollection)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?
                } else {
                    geos::Geometry::create_geometry_collection(results)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?
                };

                // Write to WKB
                let wkb = collection.to_wkb().map_err(|e| {
                    DataFusionError::Execution(format!("Failed to convert to WKB: {}", e))
                })?;
                builder.append_value(&wkb);
            }
            None => builder.append_null(),
        }
        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

fn extract_scalar_i32(arg: Option<&ColumnarValue>) -> Result<Option<i32>> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let casted = arg.cast_to(&DataType::Int32, None)?;
    match &casted {
        ColumnarValue::Scalar(scalar) if scalar.is_null() => Ok(None),
        ColumnarValue::Scalar(scalar) => Ok(Some(i32::try_from(scalar.clone())?)),
        _ => Err(DataFusionError::Execution(
            "max_vertices must be scalar".to_string(),
        )),
    }
}

fn extract_scalar_f64(arg: Option<&ColumnarValue>) -> Result<Option<f64>> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let casted = arg.cast_to(&DataType::Float64, None)?;
    match &casted {
        ColumnarValue::Scalar(scalar) if scalar.is_null() => Ok(None),
        ColumnarValue::Scalar(scalar) => Ok(Some(f64::try_from(scalar.clone())?)),
        _ => Err(DataFusionError::Execution(
            "grid_size must be scalar".to_string(),
        )),
    }
}

// ========== Phase 2: Helper Functions ==========

/// Get envelope bounds (xmin, ymin, xmax, ymax) from a geometry
/// PostGIS equivalent: lwgeom_get_bbox returning GBOX
/// Uses GEOS get_x_min/max and get_y_min/max functions directly
fn get_envelope_bounds(geom: &geos::Geometry) -> Result<(f64, f64, f64, f64)> {
    let xmin = geom
        .get_x_min()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let ymin = geom
        .get_y_min()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let xmax = geom
        .get_x_max()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let ymax = geom
        .get_y_max()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    Ok((xmin, ymin, xmax, ymax))
}

/// Create a rectangular box geometry for clipping
/// PostGIS equivalent: lwpoly_construct_envelope
fn create_box_geometry(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Result<geos::Geometry> {
    // Create WKT for a rectangular polygon
    let wkt = format!(
        "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
        xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin
    );

    geos::Geometry::new_from_wkt(&wkt).map_err(|e| DataFusionError::External(Box::new(e)))
}

/// Check if geometry is a collection but NOT a MultiPoint
/// PostGIS: lwgeom_is_collection(geom) && geom->type != MULTIPOINTTYPE
fn is_collection_not_multipoint(geom: &geos::Geometry) -> Result<bool> {
    let geom_type = geom.geometry_type();

    use geos::GeometryTypes;
    Ok(matches!(
        geom_type,
        GeometryTypes::GeometryCollection
            | GeometryTypes::MultiLineString
            | GeometryTypes::MultiPolygon
    ))
}

/// Find optimal pivot coordinate in a polygon for subdivision
/// PostGIS lines 2526-2567: lwgeom_locate_between_m
fn find_optimal_pivot_in_polygon(
    poly: &geos::Geometry,
    split_ordinate_is_x: bool,
    center: f64,
    nvertices: usize,
) -> Result<f64> {
    // If there are more points in holes than in outer ring, trim holes starting from biggest
    let mut ring_to_use = 0; // 0 = exterior ring
    let mut ring_area = 0.0;

    // Get exterior ring
    let exterior_ring = poly
        .get_exterior_ring()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    let exterior_ring_points = exterior_ring
        .get_num_coordinates()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    // Check if we have more vertices in holes than in outer ring
    if nvertices >= 2 * (exterior_ring_points as usize) {
        // Find the largest hole by area
        let num_holes = poly
            .get_num_interior_rings()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        for i in 0..num_holes {
            let hole = poly
                .get_interior_ring_n(i as u32)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let hole_area = hole
                .area()
                .map_err(|e| DataFusionError::External(Box::new(e)))?
                .abs();

            if hole_area >= ring_area {
                ring_area = hole_area;
                ring_to_use = (i + 1) as usize; // interior rings are 1-indexed in our usage
            }
        }
    }

    // Get the chosen ring
    let chosen_ring = if ring_to_use == 0 {
        exterior_ring
    } else {
        poly.get_interior_ring_n((ring_to_use - 1) as u32)
            .map_err(|e| DataFusionError::External(Box::new(e)))?
    };

    // Get coordinate sequence from the ring
    // Note: LinearRing geometries use coordinate sequences, not point-by-point access
    let coord_seq = chosen_ring
        .get_coord_seq()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    let num_points = coord_seq
        .size()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    // Find the point in the ring that is closest to the center on the split axis
    let mut pivot = f64::MAX;
    let mut pivot_eps = f64::MAX;

    for i in 0..num_points {
        let pt = if split_ordinate_is_x {
            coord_seq.get_x(i)
        } else {
            coord_seq.get_y(i)
        }
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let pt_eps = (pt - center).abs();
        if pt_eps < pivot_eps {
            pivot = pt;
            pivot_eps = pt_eps;
        }
    }

    Ok(pivot)
}

// ========== Phase 3: Core Recursive Subdivision ==========

/// Core recursive subdivision logic following PostGIS lwgeom_subdivide_recursive
/// (PostGIS lines 2437-2616)
fn subdivide_recursive(
    geom: &geos::Geometry,
    dimension: i32,
    max_vertices: i32,
    depth: i32,
    results: &mut Vec<geos::Geometry>,
    grid_size: f64,
) -> Result<()> {
    // 1. Get bounding box (PostGIS line 2450-2452)
    let (mut xmin, mut ymin, mut xmax, mut ymax) = match get_envelope_bounds(geom) {
        Ok(bounds) => bounds,
        Err(_) => return Ok(()), // No bbox, skip (PostGIS line 2452)
    };

    let mut width = xmax - xmin;
    let mut height = ymax - ymin;

    // 2. Handle degenerate geometries (PostGIS lines 2464-2482)
    if width == 0.0 && height == 0.0 {
        // For 0-dimensional Point, add it if dimension matches
        if dimension == 0 {
            let geom_type = geom.geometry_type();
            if geom_type == geos::GeometryTypes::Point {
                results.push(Geom::clone(geom));
            }
        }
        return Ok(());
    }

    // Expand by FP_TOLERANCE if width or height is 0
    if width == 0.0 {
        xmax += FP_TOLERANCE;
        xmin -= FP_TOLERANCE;
        width = 2.0 * FP_TOLERANCE;
    }
    if height == 0.0 {
        ymax += FP_TOLERANCE;
        ymin -= FP_TOLERANCE;
        height = 2.0 * FP_TOLERANCE;
    }

    // 3. Handle collections - recurse into them without incrementing depth (PostGIS lines 2484-2493)
    if is_collection_not_multipoint(geom)? {
        let num_geoms = geom
            .get_num_geometries()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        for i in 0..num_geoms {
            let sub_geom = geom
                .get_geometry_n(i)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            // Convert ConstGeometry to Geometry
            let owned_geom = Geom::clone(&sub_geom);
            subdivide_recursive(&owned_geom, dimension, max_vertices, depth, results, grid_size)?;
        }
        return Ok(());
    }

    // 4. Filter by dimension (PostGIS lines 2495-2500)
    let geom_dimension = geom
        .get_num_dimensions()
        .map_err(|e| DataFusionError::External(Box::new(e)))? as i32;
    if geom_dimension < dimension {
        // Lower dimension object from clipping, ignore it
        return Ok(());
    }

    // 5. Check depth limit (PostGIS lines 2502-2508)
    if depth > MAX_DEPTH {
        results.push(Geom::clone(geom));
        return Ok(());
    }

    // 6. Count vertices (PostGIS line 2510)
    let nvertices = geom
        .get_num_coordinates()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    // Skip empties (PostGIS lines 2512-2514)
    if nvertices == 0 {
        return Ok(());
    }

    // 7. If under vertex tolerance, add it (PostGIS lines 2516-2521)
    if nvertices as i32 <= max_vertices {
        results.push(Geom::clone(geom));
        return Ok(());
    }

    // 8. Determine split ordinate and center (PostGIS lines 2523-2524)
    let split_ordinate_is_x = width > height;
    let center = if split_ordinate_is_x {
        (xmin + xmax) / 2.0
    } else {
        (ymin + ymax) / 2.0
    };

    // 9. Find optimal pivot for polygons (PostGIS lines 2526-2567)
    let mut pivot = f64::MAX;
    if geom.geometry_type() == geos::GeometryTypes::Polygon {
        pivot = find_optimal_pivot_in_polygon(geom, split_ordinate_is_x, center, nvertices)?;
    }

    // 10. Create subboxes (PostGIS lines 2568-2588)
    let subbox1_xmin = xmin;
    let subbox1_ymin = ymin;
    let mut subbox1_xmax = xmax;
    let mut subbox1_ymax = ymax;

    let mut subbox2_xmin = xmin;
    let mut subbox2_ymin = ymin;
    let subbox2_xmax = xmax;
    let subbox2_ymax = ymax;

    // Use center if pivot is invalid (PostGIS lines 2572-2573)
    if pivot == f64::MAX {
        pivot = center;
    }

    // Split the boxes (PostGIS lines 2575-2588)
    // FP_NEQUALS(A, B) = fabs((A)-(B)) > FP_TOLERANCE
    if split_ordinate_is_x {
        if (subbox1_xmax - pivot).abs() > FP_TOLERANCE && (subbox1_xmin - pivot).abs() > FP_TOLERANCE
        {
            subbox1_xmax = pivot;
            subbox2_xmin = pivot;
        } else {
            subbox1_xmax = center;
            subbox2_xmin = center;
        }
    } else {
        if (subbox1_ymax - pivot).abs() > FP_TOLERANCE && (subbox1_ymin - pivot).abs() > FP_TOLERANCE
        {
            subbox1_ymax = pivot;
            subbox2_ymin = pivot;
        } else {
            subbox1_ymax = center;
            subbox2_ymin = center;
        }
    }

    let new_depth = depth + 1;

    // 11. Clip and recurse into first subbox (PostGIS lines 2592-2603)
    {
        let subbox = create_box_geometry(subbox1_xmin, subbox1_ymin, subbox1_xmax, subbox1_ymax)?;
        let mut clipped = geom
            .intersection(&subbox)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Apply precision if grid_size >= 0 (PostGIS line 2595: lwgeom_intersection_prec)
        if grid_size >= 0.0 {
            clipped = clipped
                .set_precision(grid_size, geos::Precision::NoTopo)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
        }

        // Simplify with tolerance 0.0 (PostGIS line 2596)
        let simplified = clipped
            .simplify(0.0)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        if !simplified
            .is_empty()
            .map_err(|e| DataFusionError::External(Box::new(e)))?
        {
            subdivide_recursive(&simplified, dimension, max_vertices, new_depth, results, grid_size)?;
        }
    }

    // 12. Clip and recurse into second subbox (PostGIS lines 2604-2615)
    {
        let subbox = create_box_geometry(subbox2_xmin, subbox2_ymin, subbox2_xmax, subbox2_ymax)?;
        let mut clipped = geom
            .intersection(&subbox)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Apply precision if grid_size >= 0 (PostGIS line 2607: lwgeom_intersection_prec)
        if grid_size >= 0.0 {
            clipped = clipped
                .set_precision(grid_size, geos::Precision::NoTopo)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
        }

        // Simplify with tolerance 0.0 (PostGIS line 2608)
        let simplified = clipped
            .simplify(0.0)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        if !simplified
            .is_empty()
            .map_err(|e| DataFusionError::External(Box::new(e)))?
        {
            subdivide_recursive(&simplified, dimension, max_vertices, new_depth, results, grid_size)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn test_st_subdivide_signature(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf =
            SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );
        tester.assert_return_type(WKB_GEOMETRY);
    }

    #[rstest]
    fn test_st_subdivide_validation(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf =
            SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        // Test: max_vertices < 5 should error
        let result = tester.invoke_scalar_scalar("POINT(0 0)", 3);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("max_vertices must be >= 5"));

        // Test: max_vertices = 5 should be accepted and return a result
        let result = tester.invoke_scalar_scalar("POINT(0 0)", 5);
        assert!(result.is_ok());

        // Test: NULL geometry should return NULL
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, 10)
            .unwrap();
        assert!(result.is_null());
    }

    #[rstest]
    fn test_st_subdivide_default_params(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        // Test: Single argument (geometry only) - should use default max_vertices=256
        let udf_1_arg = SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_impl());
        let tester_1_arg = ScalarUdfTester::new(udf_1_arg.into(), vec![sedona_type.clone()]);
        tester_1_arg.assert_return_type(WKB_GEOMETRY);

        let result = tester_1_arg.invoke_scalar("POINT(0 0)");
        assert!(result.is_ok());

        // Test: NULL max_vertices should use default
        let udf_2_arg =
            SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester_2_arg = ScalarUdfTester::new(
            udf_2_arg.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        let result = tester_2_arg.invoke_scalar_scalar("POINT(0 0)", ScalarValue::Int32(None));
        assert!(result.is_ok());

        // Test: Three arguments (with grid_size)
        let udf_3_arg =
            SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_grid_size_impl());
        let tester_3_arg = ScalarUdfTester::new(
            udf_3_arg.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester_3_arg.assert_return_type(WKB_GEOMETRY);

        let result = tester_3_arg.invoke_scalar_scalar_scalar("POINT(0 0)", 10, 0.1);
        assert!(result.is_ok());
    }

    // ========== Phase 3: Subdivision Tests ==========
    // Test cases based on PostGIS regress/core/subdivide.sql

    #[rstest]
    fn test_st_subdivide_point(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        // PostGIS test #3522: A point should not be subdivided
        let result = tester.invoke_scalar_scalar("POINT(1 1)", 10).unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION(POINT(1 1))");
    }

    #[rstest]
    fn test_st_subdivide_linestring_no_subdivision(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        // A simple linestring with few vertices should not be subdivided
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 10 0, 10 10, 0 10)", 10)
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION(LINESTRING(0 0, 10 0, 10 10, 0 10))");
    }

    #[rstest]
    fn test_st_subdivide_polygon_simple(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        // A simple polygon with few vertices should not be subdivided
        let result = tester
            .invoke_scalar_scalar("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", 10)
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION(POLYGON((0 0, 10 0, 10 10, 0 10, 0 0)))");
    }

    #[rstest]
    fn test_st_subdivide_polygon_many_vertices(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        // A polygon with many vertices that needs subdivision
        // This polygon has 14 vertices, with max_vertices=5 it should be subdivided into 2 parts
        let result = tester
            .invoke_scalar_scalar(
                "POLYGON((0 0, 100 0, 100 10, 100 20, 100 30, 100 40, 100 50, 100 60, 100 70, 100 80, 100 90, 100 100, 0 100, 0 0))",
                5
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(POLYGON((0 0, 0 50, 100 50, 100 0, 0 0)), POLYGON((100 50, 0 50, 0 100, 100 100, 100 50)))"
        );
    }

    #[rstest]
    fn test_st_subdivide_polygon_complex(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        // PostGIS test case 1: Complex polygon from regress/core/subdivide.sql
        // 28-vertex polygon subdivided with max_vertices=10 should produce 5 parts
        let result = tester
            .invoke_scalar_scalar(
                "POLYGON((132 10,119 23,85 35,68 29,66 28,49 42,32 56,22 64,32 110,40 119,36 150,57 158,75 171,92 182,114 184,132 186,146 178,176 184,179 162,184 141,190 122,190 100,185 79,186 56,186 52,178 34,168 18,147 13,132 10))",
                10
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(POLYGON((85 35, 68 29, 66 28, 32 56, 22 64, 29.82608695652174 100, 119 100, 119 23, 85 35)), POLYGON((186 52, 178 34, 168 18, 147 13, 132 10, 119 23, 119 56, 186 56, 186 52)), POLYGON((185 79, 186 56, 119 56, 119 100, 190 100, 185 79)), POLYGON((40 119, 36 150, 57 158, 75 171, 92 182, 114 184, 114 100, 29.82608695652174 100, 32 110, 40 119)), POLYGON((132 186, 146 178, 176 184, 179 162, 184 141, 190 122, 190 100, 114 100, 114 184, 132 186)))"
        );
    }

    // ========== Phase 2 Helper Function Tests ==========

    #[test]
    fn test_get_envelope_bounds_polygon() {
        let geom = geos::Geometry::new_from_wkt("POLYGON((0 0, 10 0, 10 5, 0 5, 0 0))").unwrap();
        let (xmin, ymin, xmax, ymax) = get_envelope_bounds(&geom).unwrap();
        assert_eq!(xmin, 0.0);
        assert_eq!(ymin, 0.0);
        assert_eq!(xmax, 10.0);
        assert_eq!(ymax, 5.0);
    }

    #[test]
    fn test_get_envelope_bounds_point() {
        let geom = geos::Geometry::new_from_wkt("POINT(5 10)").unwrap();
        let (xmin, ymin, xmax, ymax) = get_envelope_bounds(&geom).unwrap();
        assert_eq!(xmin, 5.0);
        assert_eq!(ymin, 10.0);
        assert_eq!(xmax, 5.0);
        assert_eq!(ymax, 10.0);
    }

    #[test]
    fn test_get_envelope_bounds_linestring() {
        let geom = geos::Geometry::new_from_wkt("LINESTRING(1 1, 5 9, 10 2)").unwrap();
        let (xmin, ymin, xmax, ymax) = get_envelope_bounds(&geom).unwrap();
        assert_eq!(xmin, 1.0);
        assert_eq!(ymin, 1.0);
        assert_eq!(xmax, 10.0);
        assert_eq!(ymax, 9.0);
    }

    #[test]
    fn test_create_box_geometry() {
        let box_geom = create_box_geometry(0.0, 0.0, 10.0, 5.0).unwrap();
        let wkt = box_geom.to_wkt().unwrap();
        assert!(wkt.contains("POLYGON"));

        // Verify the box has correct bounds
        let (xmin, ymin, xmax, ymax) = get_envelope_bounds(&box_geom).unwrap();
        assert_eq!(xmin, 0.0);
        assert_eq!(ymin, 0.0);
        assert_eq!(xmax, 10.0);
        assert_eq!(ymax, 5.0);
    }

    #[test]
    fn test_is_collection_not_multipoint() {
        // GeometryCollection should return true
        let gc =
            geos::Geometry::new_from_wkt("GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1))")
                .unwrap();
        assert!(is_collection_not_multipoint(&gc).unwrap());

        // MultiLineString should return true
        let mls = geos::Geometry::new_from_wkt("MULTILINESTRING((0 0, 1 1), (2 2, 3 3))").unwrap();
        assert!(is_collection_not_multipoint(&mls).unwrap());

        // MultiPolygon should return true
        let mp = geos::Geometry::new_from_wkt("MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)))").unwrap();
        assert!(is_collection_not_multipoint(&mp).unwrap());

        // MultiPoint should return false
        let multipoint = geos::Geometry::new_from_wkt("MULTIPOINT(0 0, 1 1)").unwrap();
        assert!(!is_collection_not_multipoint(&multipoint).unwrap());

        // Point should return false
        let point = geos::Geometry::new_from_wkt("POINT(0 0)").unwrap();
        assert!(!is_collection_not_multipoint(&point).unwrap());

        // LineString should return false
        let linestring = geos::Geometry::new_from_wkt("LINESTRING(0 0, 1 1)").unwrap();
        assert!(!is_collection_not_multipoint(&linestring).unwrap());

        // Polygon should return false
        let polygon = geos::Geometry::new_from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap();
        assert!(!is_collection_not_multipoint(&polygon).unwrap());
    }
}
