use crate::simulation::SimState;
use anyhow::Result;
use plotters::prelude::*;

pub fn draw_temperature_map(state: &SimState, output_path: &str) -> Result<()> {
    let (ny, nx) = state.temp.dim();
    let root = BitMapBackend::new(output_path, (800, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .title("Temperature Distribution", ("sans-serif", 30))
        .margin(20)
        .build_cartesian_2d(0..nx, 0..ny)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    // Create a heatmap
    chart.draw_series(
        (0..nx).flat_map(|x| (0..ny).map(move |y| (x, y, state.temp[[y, x]])))
        .map(|(x, y, temp)| {
            let color = HSLColor(240.0 * (1.0 - temp), 0.7, 0.5); // Blue (cold) to Red (hot)
            Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
        })
    )?;

    // Add a color bar
    chart.configure_series_labels().border_style(&BLACK).draw()?;
    
    let mut color_bar_chart = ChartBuilder::on(&root)
        .margin_left(700)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..1, 0f64..1f64)?;

    color_bar_chart.draw_series(
        (0..100).map(|i| i as f64 / 100.0)
        .map(|y| {
            let color = HSLColor(240.0 * (1.0 - y), 0.7, 0.5);
            Rectangle::new([(0, y), (1, y + 0.01)], color.filled())
        })
    )?;

    root.present()?;
    println!("Visualization saved to {}", output_path);
    Ok(())
}