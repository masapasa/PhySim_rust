
// gn_pinn.rs
use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor, Device};
use std::fs::OpenOptions;
use std::io::Write;

mod residuals;
use residuals::compute_pde_residuals;

fn main() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = &vs.root();

    // Build a simple PINN model: input [x, y, t], output [u, v, p, T, phi]
    let model = nn::seq()
        .add(nn::linear(root / "layer1", 3, 64, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(root / "layer2", 64, 64, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(root / "layer3", 64, 64, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(root / "layer4", 64, 5, Default::default()));

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    // Sample input points (x, y, t)
    let xy = Tensor::rand(&[1000, 3], (tch::Kind::Float, device));

    // Initialize CSV logging
    let mut log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("training_log.csv")
        .expect("Unable to create log file");
    writeln!(log_file, "epoch,loss").unwrap();

    for epoch in 0..10000 {
        let residuals = compute_pde_residuals(&model, &xy);
        let loss = residuals.pow(2).mean(tch::Kind::Float);

        opt.zero_grad();
        loss.backward();
        opt.step();

        if epoch % 100 == 0 {
            let loss_val = f64::from(&loss);
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
            writeln!(log_file, "{epoch},{:.6}", loss_val).unwrap();

            // Interface tracking: log (x, y, t) where phi ≈ 0.5 ± 0.05
            let pred = model.forward(&xy);
            let phi = pred.narrow(1, 4, 1);
            let mask = (phi.ge(0.45) * phi.le(0.55)).squeeze();
            let front_pts = xy.shallow_clone().index_select(0, &mask.nonzero().squeeze());

            if front_pts.size()[0] > 0 {
                let mut front_file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("interface_points.csv")
                    .expect("Unable to open interface log");

                for i in 0..front_pts.size()[0] {
                    let x = f64::from(&front_pts.get(i).get(0));
                    let y = f64::from(&front_pts.get(i).get(1));
                    let t = f64::from(&front_pts.get(i).get(2));
                    writeln!(front_file, "{epoch},{x:.5},{y:.5},{t:.5}").unwrap();
                }
            }
        }
    }

    // Save model (optional)
    vs.save("model.ot").unwrap();
}
