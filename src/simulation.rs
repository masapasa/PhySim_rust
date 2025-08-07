use ndarray::{Array, Array2, Axis};

// Holds the parameters for a simulation run.
pub struct SimParameters {
    pub nx: usize, // Number of grid points in x
    pub ny: usize, // Number of grid points in y
    pub dt: f64,   // Time step
    pub pr: f64,   // Prandtl number
    pub ra: f64,   // Rayleigh number
}

// Holds the state of the simulation at a given time.
pub struct SimState {
    pub temp: Array2<f64>,        // Temperature
    pub vort: Array2<f64>,        // Vorticity
    pub stream: Array2<f64>,      // Stream function
    pub u: Array2<f64>,           // X-velocity
    pub v: Array2<f64>,           // Y-velocity
}

impl SimState {
    pub fn new(nx: usize, ny: usize) -> Self {
        SimState {
            temp: Array::zeros((ny, nx)),
            vort: Array::zeros((ny, nx)),
            stream: Array::zeros((ny, nx)),
            u: Array::zeros((ny, nx)),
            v: Array::zeros((ny, nx)),
        }
    }
}

// Main simulation controller.
pub struct Simulation {
    pub params: SimParameters,
    pub state: SimState,
    dx: f64,
    dy: f64,
}

impl Simulation {
    pub fn new(params: SimParameters) -> Self {
        let mut sim = Simulation {
            dx: 1.0 / (params.nx as f64 - 1.0),
            dy: 1.0 / (params.ny as f64 - 1.0),
            state: SimState::new(params.nx, params.ny),
            params,
        };
        sim.initialize_conditions();
        sim
    }

    // Set initial and boundary conditions.
    fn initialize_conditions(&mut self) {
        let (ny, nx) = self.state.temp.dim();
        // Boundary Conditions: Hot bottom/sides, cold top (the "crystal")
        self.state.temp.row_mut(0).fill(1.0); // Bottom wall
        self.state.temp.row_mut(ny - 1).fill(0.0); // Top wall
        self.state.temp.column_mut(0).fill(1.0); // Left wall
        self.state.temp.column_mut(nx-1).fill(1.0); // Right wall
    }

    // Perform one time step.
    pub fn step(&mut self) {
        let (ny, nx) = self.state.temp.dim();
        let dx = self.dx;
        let dy = self.dy;
        let dt = self.params.dt;

        let mut temp_new = self.state.temp.clone();
        let mut vort_new = self.state.vort.clone();

        // 1. Solve Stream Function (Poisson equation: ∇²ψ = -ω) using Jacobi iteration
        for _ in 0..50 { // Iterate to converge
            let stream_old = self.state.stream.clone();
            for i in 1..ny - 1 {
                for j in 1..nx - 1 {
                    self.state.stream[[i, j]] = ((stream_old[[i, j+1]] + stream_old[[i, j-1]]) * dy*dy +
                                                (stream_old[[i+1, j]] + stream_old[[i-1, j]]) * dx*dx -
                                                self.state.vort[[i, j]] * dx*dx*dy*dy) /
                                                (2.0 * (dx*dx + dy*dy));
                }
            }
        }

        // 2. Update Velocities (u = ∂ψ/∂y, v = -∂ψ/∂x)
        for i in 1..ny-1 {
            for j in 1..nx-1 {
                self.state.u[[i,j]] = (self.state.stream[[i+1, j]] - self.state.stream[[i-1, j]]) / (2.0*dy);
                self.state.v[[i,j]] = -(self.state.stream[[i, j+1]] - self.state.stream[[i, j-1]]) / (2.0*dx);
            }
        }

        // 3. Update Boundary Vorticity (Thom's formula for no-slip walls)
        for j in 1..nx-1 {
            // Bottom wall
            self.state.vort[[0, j]] = -2.0 * self.state.stream[[1, j]] / (dy*dy);
            // Top wall
            self.state.vort[[ny-1, j]] = -2.0 * self.state.stream[[ny-2, j]] / (dy*dy);
        }
        for i in 1..ny-1 {
            // Left wall
            self.state.vort[[i, 0]] = -2.0 * self.state.stream[[i, 1]] / (dx*dx);
            // Right wall
            self.state.vort[[i, nx-1]] = -2.0 * self.state.stream[[i, nx-2]] / (dx*dx);
        }

        // 4. Time-step Vorticity and Temperature (Advection-Diffusion equations)
        for i in 1..ny-1 {
            for j in 1..nx-1 {
                // Advection terms (using simple upwinding for stability)
                let u = self.state.u[[i,j]];
                let v = self.state.v[[i,j]];

                let vort_adv_x = if u > 0.0 { u * (self.state.vort[[i,j]] - self.state.vort[[i,j-1]]) / dx } else { u * (self.state.vort[[i,j+1]] - self.state.vort[[i,j]]) / dx };
                let vort_adv_y = if v > 0.0 { v * (self.state.vort[[i,j]] - self.state.vort[[i-1,j]]) / dy } else { v * (self.state.vort[[i,j+1]] - self.state.vort[[i,j]]) / dy };

                let temp_adv_x = if u > 0.0 { u * (self.state.temp[[i,j]] - self.state.temp[[i,j-1]]) / dx } else { u * (self.state.temp[[i,j+1]] - self.state.temp[[i,j]]) / dx };
                let temp_adv_y = if v > 0.0 { v * (self.state.temp[[i,j]] - self.state.temp[[i-1,j]]) / dy } else { v * (self.state.temp[[i,j+1]] - self.state.temp[[i,j]]) / dy };


                // Diffusion terms
                let vort_diff = self.params.pr * ( (self.state.vort[[i, j+1]] - 2.0*self.state.vort[[i,j]] + self.state.vort[[i,j-1]])/(dx*dx) + (self.state.vort[[i+1, j]] - 2.0*self.state.vort[[i,j]] + self.state.vort[[i-1, j]])/(dy*dy) );
                let temp_diff = (self.state.temp[[i, j+1]] - 2.0*self.state.temp[[i,j]] + self.state.temp[[i,j-1]])/(dx*dx) + (self.state.temp[[i+1, j]] - 2.0*self.state.temp[[i,j]] + self.state.temp[[i-1, j]])/(dy*dy);
                
                // Buoyancy term for vorticity
                let buoyancy = self.params.ra * self.params.pr * (self.state.temp[[i, j+1]] - self.state.temp[[i, j-1]]) / (2.0 * dx);

                // Update using Forward Euler
                vort_new[[i, j]] = self.state.vort[[i, j]] + dt * (vort_diff - vort_adv_x - vort_adv_y + buoyancy);
                temp_new[[i, j]] = self.state.temp[[i, j]] + dt * (temp_diff - temp_adv_x - temp_adv_y);
            }
        }
        self.state.vort = vort_new;
        self.state.temp = temp_new;
    }

    // Run the full simulation.
    pub fn run(&mut self, time_steps: usize) {
        for step in 0..time_steps {
            self.step();
            if step % 100 == 0 {
                println!("Completed step {}/{}", step, time_steps);
            }
        }
    }
}