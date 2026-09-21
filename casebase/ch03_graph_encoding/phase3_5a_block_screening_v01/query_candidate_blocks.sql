SELECT
    city,
    block_id,
    area_m2,
    elongation,
    compactness,
    lod1_building_count,
    bcr,
    convexity,
    rectangularity,
    far,
    bcd,
    osr,
    avh,
    avh_w,
    hsd,
    hcv,
    h_max,
    h_range,
    total_volume,
    avg_volume,
    avg_footprint,
    footprint_cv,
    avg_complexity,
    avg_rectangularity,
    avg_convexity,
    avg_orientation,
    orientation_entropy,
    avg_nn_distance,
    geom IS NULL AS geom_is_null,
    CASE WHEN geom IS NULL THEN TRUE ELSE ST_IsEmpty(geom) END AS geom_is_empty,
    CASE WHEN geom IS NULL THEN FALSE ELSE ST_IsValid(geom) END AS geom_is_valid,
    CASE
        WHEN geom IS NULL OR ST_IsEmpty(geom) OR NOT ST_IsValid(geom) THEN FALSE
        ELSE ST_Area(geom::geography) > 0
    END AS geom_area_positive
FROM summary.lod1_candidate_blocks
ORDER BY city, block_id;
