use clap::{Parser, Subcommand};
use anyhow::Result;

mod db;
mod models;
mod schema;
mod simulation;
mod visualization;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a new simulation and save the results
    Run {
        #[arg(short, long, default_value = "Test Run")]
        description: String,
        #[arg(short, long, default_value_t = 41)]
        grid_size: usize,
        #[arg(short, long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 0.71)]
        prandtl: f64,
        #[arg(long, default_value_t = 10000.0)]
        rayleigh: f64,
    },
    /// List all previous simulation runs
    List,
    /// Query a past simulation and generate a visualization
    Query {
        #[arg(short, long)]
        id: i32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let pool = db::establish_connection_pool();

    match &cli.command {
        Commands::Run { description, grid_size, steps, prandtl, rayleigh } => {
            println!("Starting new simulation...");

            // 1. Create a record for this simulation run
            let run = db::create_simulation_run(
                &pool,
                description,
                *grid_size as i32,
                *steps as i32,
                *prandtl,
                *rayleigh,
            )?;
            println!("Created simulation run with ID: {}", run.id);

            // 2. Setup and run the simulation
            let params = simulation::SimParameters {
                nx: *grid_size,
                ny: *grid_size,
                dt: 0.0001, // A small, stable time step
                pr: *prandtl,
                ra: *rayleigh,
            };
            let mut sim = simulation::Simulation::new(params);
            sim.run(*steps);

            // 3. Save the results to the database
            println!("Saving results to database...");
            db::save_simulation_results(&pool, run.id, &sim.state)?;
            println!("Results saved successfully.");

            // 4. Generate a visualization
            let output_file = format!("run_{}_temp.png", run.id);
            visualization::draw_temperature_map(&sim.state, &output_file)?;
        }
        Commands::List => {
            println!("Querying simulation runs from the database...");
            db::list_simulation_runs(&pool)?;
        }
        Commands::Query { id } => {
            println!("Querying results for run ID: {}", id);
            let state = db::get_simulation_results(&pool, *id)?;
            println!("Results retrieved. Generating visualization...");
            
            let output_file = format!("queried_run_{}_temp.png", id);
            visualization::draw_temperature_map(&state, &output_file)?;
        }
    }

    Ok(())
}