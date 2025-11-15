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
    _grid_size: f64,
) -> Result<ColumnarValue> {
    // TODO: grid_size will be used in Phase 3 for precision grid snapping

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
    _poly: &geos::Geometry,
    _split_ordinate_is_x: bool,
    _center: f64,
    _nvertices: usize,
) -> Result<f64> {
    // TODO: This will be implemented when needed in Phase 3
    // For now, just use the center value
    Ok(_center)
}

// ========== Phase 3: Core Recursive Subdivision ==========

/// Core recursive subdivision logic following PostGIS lwgeom_subdivide_recursive
/// (PostGIS lines 2437-2616)
fn subdivide_recursive(
    _geom: &geos::Geometry,
    _dimension: i32,
    _max_vertices: i32,
    _depth: i32,
    _results: &mut Vec<geos::Geometry>,
) -> Result<()> {
    // TODO: Implementation will be added in Phase 3
    Err(DataFusionError::NotImplemented(
        "subdivide_recursive not yet implemented".to_string(),
    ))
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

        // Test: max_vertices = 5 should be accepted (will fail on not implemented, but validation passes)
        let result = tester.invoke_scalar_scalar("POINT(0 0)", 5);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not yet implemented"));

        // Test: NULL geometry should return NULL
        let result = tester.invoke_scalar_scalar(ScalarValue::Null, 10).unwrap();
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
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not yet implemented"));

        // Test: NULL max_vertices should use default
        let udf_2_arg =
            SedonaScalarUDF::from_kernel("st_subdivide", st_subdivide_with_max_vertices_impl());
        let tester_2_arg = ScalarUdfTester::new(
            udf_2_arg.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int32)],
        );

        let result = tester_2_arg.invoke_scalar_scalar("POINT(0 0)", ScalarValue::Int32(None));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not yet implemented"));

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
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not yet implemented"));
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
