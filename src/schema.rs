// @generated automatically by Diesel CLI.

diesel::table! {
    results (id) {
        id -> Int8,
        run_id -> Int4,
        x -> Int4,
        y -> Int4,
        temperature -> Float8,
        u_velocity -> Float8,
        v_velocity -> Float8,
    }
}

diesel::table! {
    simulation_runs (id) {
        id -> Int4,
        description -> Text,
        grid_size -> Int4,
        time_steps -> Int4,
        prandtl_number -> Float8,
        rayleigh_number -> Float8,
        created_at -> Timestamp,
    }
}

diesel::joinable!(results -> simulation_runs (run_id));

diesel::allow_tables_to_appear_in_same_query!(
    results,
    simulation_runs,
);