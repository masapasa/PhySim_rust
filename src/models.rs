use crate::schema::{results, simulation_runs};
use diesel::prelude::*;
use chrono::NaiveDateTime;

#[derive(Queryable, Identifiable, Selectable, Debug)]
#[diesel(table_name = simulation_runs)]
pub struct SimulationRun {
    pub id: i32,
    pub description: String,
    pub grid_size: i32,
    pub time_steps: i32,
    pub prandtl_number: f64,
    pub rayleigh_number: f64,
    pub created_at: NaiveDateTime,
}

#[derive(Insertable)]
#[diesel(table_name = simulation_runs)]
pub struct NewSimulationRun<'a> {
    pub description: &'a str,
    pub grid_size: i32,
    pub time_steps: i32,
    pub prandtl_number: f64,
    pub rayleigh_number: f64,
}

#[derive(Insertable)]
#[diesel(table_name = results)]
pub struct NewResultPoint {
    pub run_id: i32,
    pub x: i32,
    pub y: i32,
    pub temperature: f64,
    pub u_velocity: f64,
    pub v_velocity: f64,
}