use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::r2d2::{self, ConnectionManager};
use dotenvy::dotenv;
use std::env;
use anyhow::Result;

use crate::models::{NewResultPoint, NewSimulationRun, SimulationRun};
use crate::schema::{results, simulation_runs};
use crate::simulation::SimState;

pub type DbPool = r2d2::Pool<ConnectionManager<PgConnection>>;

pub fn establish_connection_pool() -> DbPool {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let manager = ConnectionManager::<PgConnection>::new(database_url);
    r2d2::Pool::builder()
        .build(manager)
        .expect("Failed to create pool.")
}

pub fn create_simulation_run(
    pool: &DbPool,
    desc: &str,
    grid_size: i32,
    time_steps: i32,
    pr: f64,
    ra: f64,
) -> Result<SimulationRun> {
    let mut conn = pool.get()?;
    let new_run = NewSimulationRun {
        description: desc,
        grid_size,
        time_steps,
        prandtl_number: pr,
        rayleigh_number: ra,
    };

    let run = diesel::insert_into(simulation_runs::table)
        .values(&new_run)
        .get_result(&mut conn)?;
    Ok(run)
}

pub fn save_simulation_results(
    pool: &DbPool,
    run_id: i32,
    final_state: &SimState,
) -> Result<()> {
    let mut conn = pool.get()?;
    let (ny, nx) = final_state.temp.dim();
    let mut new_points = Vec::new();

    for i in 0..ny {
        for j in 0..nx {
            new_points.push(NewResultPoint {
                run_id,
                x: j as i32,
                y: i as i32,
                temperature: final_state.temp[[i, j]],
                u_velocity: final_state.u[[i, j]],
                v_velocity: final_state.v[[i, j]],
            });
        }
    }

    // Use a bulk insert for performance
    diesel::insert_into(results::table)
        .values(&new_points)
        .execute(&mut conn)?;

    Ok(())
}

pub fn list_simulation_runs(pool: &DbPool) -> Result<()> {
    use crate::schema::simulation_runs::dsl::*;
    let mut conn = pool.get()?;
    let runs = simulation_runs.load::<SimulationRun>(&mut conn)?;

    println!("--- Available Simulation Runs ---");
    println!(
        "{:<5} | {:<25} | {:<10} | {:<10} | {:<10} | {:<10}",
        "ID", "Description", "Grid", "Steps", "Pr", "Ra"
    );
    println!("{}", "-".repeat(85));
    for run in runs {
        println!(
            "{:<5} | {:<25} | {:<10} | {:<10} | {:<10.2} | {:<10.1e}",
            run.id, run.description, run.grid_size, run.time_steps, run.prandtl_number, run.rayleigh_number
        );
    }
    Ok(())
}


pub fn get_simulation_results(pool: &DbPool, run_id_to_get: i32) -> Result<SimState> {
    use crate::schema::results::dsl::*;
    use crate::schema::simulation_runs::dsl::{grid_size, simulation_runs};

    let mut conn = pool.get()?;

    // First get grid size for the run
    let size = simulation_runs
        .find(run_id_to_get)
        .select(grid_size)
        .first::<i32>(&mut conn)?;
    
    let (nx, ny) = (size as usize, size as usize);
    let mut state = SimState::new(nx, ny);

    // Get all result points for the run
    let points = results
        .filter(run_id.eq(run_id_to_get))
        .load::<(i64, i32, i32, i32, f64, f64, f64)>(&mut conn)?;

    for point in points {
        let (_id, _run_id, x, y, temp, u_vel, v_vel) = point;
        if x < nx as i32 && y < ny as i32 {
            state.temp[[y as usize, x as usize]] = temp;
            state.u[[y as usize, x as usize]] = u_vel;
            state.v[[y as usize, x as usize]] = v_vel;
        }
    }

    Ok(state)
}