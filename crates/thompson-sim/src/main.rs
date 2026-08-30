//! Regret and throughput harness.
//!
//! ```text
//! cargo run --release -p thompson-sim -- --seeds 30
//! cargo run --release -p thompson-sim -- --group sampler --scenario hard
//! cargo run --release -p thompson-sim -- --csv results.csv
//! ```

use std::fmt::Write as _;
use thompson_sim::env;
use thompson_sim::experiment::{evaluate, Summary};
use thompson_sim::treatments;

struct Args {
    seeds: usize,
    group: Option<String>,
    scenario: Option<String>,
    csv: Option<String>,
    list: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        seeds: 20,
        group: None,
        scenario: None,
        csv: None,
        list: false,
    };

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || {
            argv.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--seeds" => {
                args.seeds = value()?
                    .parse()
                    .map_err(|_| "--seeds must be a positive integer".to_string())?;
                if args.seeds == 0 {
                    return Err("--seeds must be at least 1".to_string());
                }
            }
            "--group" => args.group = Some(value()?),
            "--scenario" => args.scenario = Some(value()?),
            "--csv" => args.csv = Some(value()?),
            "--list" => args.list = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unrecognised argument: {other}")),
        }
    }

    Ok(args)
}

fn usage() -> String {
    let groups: Vec<&str> = treatments::all_groups()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let scenarios: Vec<&str> = env::scenarios().into_iter().map(|s| s.name).collect();
    format!(
        "thompson-sim — regret and throughput harness\n\
         \n\
         Options:\n  \
           --seeds N        independent runs per cell (default 20)\n  \
           --group NAME     restrict to one group: {}\n  \
           --scenario NAME  restrict to one scenario: {}\n  \
           --csv PATH       also write results as CSV\n  \
           --list           describe scenarios and exit\n",
        groups.join(", "),
        scenarios.join(", ")
    )
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("error: {msg}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    let scenarios = env::scenarios();

    if args.list {
        println!("Scenarios\n");
        for s in &scenarios {
            println!(
                "  {:<8} {:>6} rounds, {} arms — {}",
                s.name,
                s.horizon,
                s.arms.len(),
                s.description
            );
        }
        return;
    }

    if let Some(name) = &args.scenario {
        if !scenarios.iter().any(|s| s.name == name) {
            eprintln!("error: unknown scenario '{name}'\n\n{}", usage());
            std::process::exit(2);
        }
    }

    let all = treatments::all_groups();
    if let Some(name) = &args.group {
        if !all.iter().any(|(g, _)| g == name) {
            eprintln!("error: unknown group '{name}'\n\n{}", usage());
            std::process::exit(2);
        }
    }

    let mut summaries: Vec<Summary> = Vec::new();
    let mut printed_any = false;

    for (group_name, group) in &all {
        if args.group.as_deref().is_some_and(|g| g != *group_name) {
            continue;
        }

        for scenario_name in treatments::scenarios_for(group_name) {
            if args
                .scenario
                .as_deref()
                .is_some_and(|s| s != *scenario_name)
            {
                continue;
            }
            let scenario = scenarios
                .iter()
                .find(|s| s.name == *scenario_name)
                .expect("scenario list is validated at startup");

            let cells: Vec<Summary> = group
                .iter()
                .map(|t| evaluate(scenario, t, args.seeds))
                .collect();

            print_table(group_name, scenario.name, scenario.description, &cells);
            summaries.extend(cells);
            printed_any = true;
        }
    }

    if !printed_any {
        eprintln!(
            "error: that group and scenario combination has nothing to run.\n\
             Groups are only run against scenarios that can distinguish them; \
             see --list."
        );
        std::process::exit(2);
    }

    println!(
        "\n{} seeds per cell. Regret is cumulative and lower is better; \
         the interval is 95% on the mean.",
        args.seeds
    );

    if let Some(path) = &args.csv {
        match write_csv(path, &summaries) {
            Ok(()) => println!("Wrote {path}"),
            Err(e) => {
                eprintln!("error: could not write {path}: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn print_table(group: &str, scenario: &str, description: &str, cells: &[Summary]) {
    println!("\n{group} · {scenario}");
    println!("{description}");

    let baseline = cells
        .iter()
        .map(|c| c.mean_regret)
        .fold(f64::INFINITY, f64::min);

    let headers = [
        "treatment",
        "regret",
        "95% CI",
        "vs best",
        "optimal",
        "ns/sel",
    ];
    let rows: Vec<[String; 6]> = cells
        .iter()
        .map(|c| {
            let relative = if baseline > 0.0 {
                format!("{:.2}x", c.mean_regret / baseline)
            } else {
                "—".to_string()
            };
            [
                c.treatment.clone(),
                format!("{:.1}", c.mean_regret),
                if c.regret_ci95().is_nan() {
                    "—".to_string()
                } else {
                    format!("±{:.1}", c.regret_ci95())
                },
                relative,
                format!("{:.1}%", c.mean_optimal_share * 100.0),
                format!("{:.0}", c.nanos_per_decision),
            ]
        })
        .collect();

    let mut widths = headers.map(str::len);
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    let mut out = String::new();
    for (i, h) in headers.iter().enumerate() {
        let _ = write!(out, "  {:<width$}", h, width = widths[i]);
    }
    out.push('\n');
    for (i, _) in headers.iter().enumerate() {
        let _ = write!(out, "  {}", "-".repeat(widths[i]));
    }
    for row in &rows {
        out.push('\n');
        for (i, cell) in row.iter().enumerate() {
            let _ = write!(out, "  {:<width$}", cell, width = widths[i]);
        }
    }
    println!("{out}");
}

fn write_csv(path: &str, summaries: &[Summary]) -> std::io::Result<()> {
    let mut out = String::from(
        "group,scenario,treatment,seeds,mean_regret,stderr_regret,ci95,optimal_share,nanos_per_decision\n",
    );
    for s in summaries {
        let _ = writeln!(
            out,
            "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.1}",
            s.group,
            s.scenario,
            s.treatment,
            s.seeds,
            s.mean_regret,
            s.stderr_regret,
            s.regret_ci95(),
            s.mean_optimal_share,
            s.nanos_per_decision
        );
    }
    std::fs::write(path, out)
}
