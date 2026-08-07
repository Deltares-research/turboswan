#!/usr/bin/env python
"""
ML Wave model validation against benchmark SWAN and observations

This script benchmarks Machine Learning model (TurboSWAN)
against SWAN-DCSM and observations.

Set-up a config function like for example (see the function in this script):
def v0002_v0003_v0004_v0005_v0006_v0007_v0008_v0009_v0010_v0011_v0012_v0014()

and run it in main wit the desired output figures (barplots, scatters, timeseries,
map snapshots, RMSE maps etc.)

Author: Elias de Korte
Date: 2025
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from turboswanvalid.core import WaveValidationExperiment
from turboswanvalid.utils import create_experiment_config


def create_custom_experiment_config(
    experiment_name: str,
    base_data_path: str,
    time_period: Tuple[str, str],
    variables: List[str],
    stations: List[str],
    selected_variants: Optional[List[str]] = None,
    selected_ensemble_members: Optional[List[int]] = None,
):
    """
    Create experiment configuration with ensemble-matched SWAN models.

    Creates separate SWAN models for each ensemble member so each ML variant
    is compared against SWAN data from the same ensemble member.

    Parameters
    ----------
    experiment_name : str
        Name for the experiment
    base_data_path : str
        Base directory path
    time_period : Tuple[str, str]
        Start and end dates
    variables : List[str]
        Variables to analyze
    stations : List[str]
        Stations to analyze
    selected_variants : List[str], optional
        Specific TurboSWAN variants to include (e.g., ['v0003'])
    selected_ensemble_members : List[int], optional
        Specific ensemble members to include (e.g., [26, 27, 28])

    Returns
    -------
    ExperimentConfig
        Custom configuration with ensemble-matched models
    """
    from turboswanvalid.experiments import ExperimentConfig, ModelConfig

    base_path = Path(base_data_path)

    if selected_variants is None:
        selected_variants = ["v0003"]

    if selected_ensemble_members is None:
        selected_ensemble_members = [26, 27, 28, 29, 30]

    # Configure main SWAN physics model (reference)
    physics_model = ModelConfig(
        name="SWAN",
        data_path=base_path / "guus/trained_models/turbo_swan_v0000",
        filename="turboswan-v0000-ens28-test-predictions.nc",
        model_type="physics",
        display_name="SWAN",
    )

    # Configure TurboSWAN base model
    ml_base_model = ModelConfig(
        name="TurboSWAN_base",
        data_path=base_path / "guus/trained_models/turbo_swan_v0000",
        filename="turboswan-v0000-ens29-test-predictions.nc",
        model_type="ml_base",
        display_name="TurboSWAN Base",
    )

    # Configure selected ML variants AND their corresponding SWAN models
    ml_variants = {}

    for variant in selected_variants:
        for ens_num in selected_ensemble_members:
            # ML variant
            variant_name = f"TurboSWAN_{variant}_ens{ens_num}"
            ml_variants[variant_name] = ModelConfig(
                name=variant_name,
                data_path=base_path / f"guus/trained_models/turbo_swan_{variant}",
                filename=f"turboswan-{variant}-ens{ens_num}-test-predictions.nc",
                model_type="ml_variant",
                display_name=f"TurboSWAN {variant.upper()} Ens{ens_num}",
            )

            # CORRESPONDING SWAN model from the same file
            swan_name = f"SWAN_{variant}_ens{ens_num}"
            ml_variants[swan_name] = ModelConfig(
                name=swan_name,
                data_path=base_path / f"guus/trained_models/turbo_swan_{variant}",
                filename=f"turboswan-{variant}-ens{ens_num}-test-predictions.nc",
                model_type="physics",
                display_name=f"SWAN {variant.upper()} Ens{ens_num}",
            )

    return ExperimentConfig(
        name=experiment_name,
        time_period=time_period,
        variables=variables,
        stations=stations,
        physics_model=physics_model,
        ml_base_model=ml_base_model,
        ml_variants=ml_variants,
        observations_path=base_path / "dev/swanprobpy/data/raw/observations/data_2022",
        output_path=base_path
        / f"guus/trained_models/turbo_swan_v0017/figures/{experiment_name}",
    )


def run_validation_analysis(
    experiment_name: str = "validation_analysis",
    base_data_path: str = "/gpfs/work2/0/prjs1027",
    time_period: Tuple[str, str] = ("1994-01-01", "1995-01-01"),
    variables: List[str] = ["Hsig"],
    stations: List[str] = ["A122", "K13a"],
    selected_variants: List[str] = ["v0003"],
    selected_ensemble_members: List[int] = [26, 27],
    # use_ensemble_specific_swan: bool = True,  # NEW PARAMETER
    create_timeseries: bool = True,
    create_scatter: bool = True,
    create_gridded_scatter: bool = False,
    create_rmse_maps: bool = True,
    create_snapshots: bool = False,
    create_ensemble_bar_plots: bool = True,
    snapshot_timestamps: Optional[List[str]] = None,
    calculate_metrics: bool = True,
) -> Dict[str, any]:
    """
    Run wave model validation analysis with option to use ensemble-specific SWAN data.

    Parameters
    ----------
    use_ensemble_specific_swan : bool, default True
        If True, compares each ML variant against its corresponding SWAN ensemble member
        If False, compares all variants against the original v0000 ens29 SWAN data
    """
    print(f"Wave Model Validation with Comprehensive Analysis")
    print(f"=================================================")
    print(f"Experiment: {experiment_name}")
    print(f"Time period: {time_period[0]} to {time_period[1]}")
    print(f"Variables: {', '.join(variables)}")
    print(f"Stations: {', '.join(stations)}")
    print(f"Variants: {', '.join(selected_variants)}")
    print(f"Ensemble members: {selected_ensemble_members}")
    # print(f"Use ensemble-specific SWAN: {use_ensemble_specific_swan}")

    # Create custom configuration with ensemble-specific SWAN option
    config = create_custom_experiment_config(
        experiment_name=experiment_name,
        base_data_path=base_data_path,
        time_period=time_period,
        variables=variables,
        stations=stations,
        selected_variants=selected_variants,
        selected_ensemble_members=selected_ensemble_members,
        # use_ensemble_specific_swan=use_ensemble_specific_swan  # Pass the parameter
    )

    print(f"Output directory: {config.output_path}")
    print(f"Total models configured: {len(config.all_models)}")

    # Rest of the function remains the same...
    # Create validation experiment
    experiment = WaveValidationExperiment(config)

    results = {
        "timeseries_plots": [],
        "scatter_plots": [],
        "gridded_scatter_plots": [],
        "rmse_maps": [],
        "snapshots": [],
        "ensemble_bar_plots": [],  # NEW RESULT CATEGORY
        "metrics_files": [],
    }

    # Models to plot
    models_to_plot = list(experiment.all_models.keys())
    ml_models = [name for name in models_to_plot if name != "SWAN"]

    print(f"\nModels to analyze: {', '.join(models_to_plot)}")

    # Create timeseries plots
    if create_timeseries:
        print(f"\nCreating timeseries plots...")
        for station in stations:
            for variable in variables:
                plot_path = experiment.create_timeseries_plot(
                    station_name=station,
                    variable=variable,
                    models_to_plot=models_to_plot,
                    include_observations=False,
                    include_scatter=False,
                    show_metrics=False,
                )
                try:
                    plot_path = experiment.create_timeseries_plot(
                        station_name=station,
                        variable=variable,
                        models_to_plot=models_to_plot,
                        include_observations=False,
                        include_scatter=False,
                        show_metrics=False,
                    )

                    results["timeseries_plots"].append(plot_path)
                    print(f"  Created: {Path(plot_path).name} in timeseries/")
                except Exception as e:
                    print(f"  Error creating timeseries for {station}, {variable}: {e}")

    # Create comprehensive scatter plots
    if create_scatter:
        print(f"\nCreating comprehensive scatter plots...")
        for variable in variables:
            try:
                from turboswanvalid.vizualization import WaveScatterPlotter

                plotter = WaveScatterPlotter(
                    "temp",
                    variable,
                    experiment.config.output_path,
                    base_output_path=experiment.config.output_path,
                )

                plot_paths = plotter.create_comprehensive_plots(
                    experiment=experiment,
                    stations=stations,
                    variable=variable,
                    plot_type="hexbin",
                )

                results["scatter_plots"].extend(plot_paths)
                print(
                    f"  Created {len(plot_paths)} comprehensive scatter plots for {variable}"
                )

            except Exception as e:
                print(
                    f"  Error creating comprehensive scatter plots for {variable}: {e}"
                )
                import traceback

                traceback.print_exc()

    # Create comprehensive gridded scatter plots
    if create_gridded_scatter:
        print(f"\nCreating comprehensive gridded scatter plots...")
        try:
            gridded_plot_paths = experiment.create_comprehensive_gridded_plots(
                variables=variables,
                time_period=time_period,
                models_to_analyze=ml_models,
            )

            results["gridded_scatter_plots"].extend(gridded_plot_paths)
            print(
                f"  Created {len(gridded_plot_paths)} comprehensive gridded scatter plots"
            )

        except Exception as e:
            print(f"  Error creating comprehensive gridded scatter plots: {e}")
            import traceback

            traceback.print_exc()

    # Create comprehensive RMSE maps
    if create_rmse_maps:
        print(f"\nCreating comprehensive spatial RMSE maps...")
        try:
            rmse_map_paths = experiment.create_comprehensive_rmse_maps(
                variables=variables,
                time_period=time_period,
                models_to_analyze=ml_models,
            )

            results["rmse_maps"].extend(rmse_map_paths)
            print(f"  Created {len(rmse_map_paths)} comprehensive RMSE maps")

        except Exception as e:
            print(f"  Error creating comprehensive RMSE maps: {e}")
            import traceback

            traceback.print_exc()

    # Create snapshot maps
    if create_snapshots and snapshot_timestamps:
        print(f"\nCreating snapshot maps...")
        try:
            snapshot_paths = experiment.create_snapshot_maps(
                variables=variables,
                timestamps=snapshot_timestamps,
                models_to_plot=models_to_plot,
                add_quiver=True,  # Add wave direction arrows
            )

            results["snapshots"].extend(snapshot_paths)
            print(f"  Created {len(snapshot_paths)} snapshot maps")

        except Exception as e:
            print(f"  Error creating snapshot maps: {e}")
            import traceback

            traceback.print_exc()

    # NEW: Create ensemble metrics bar plots
    if create_ensemble_bar_plots:
        print(f"\nCreating ensemble metrics bar plots...")
        try:
            bar_plot_paths = experiment.create_ensemble_metrics_bar_plots(
                stations=stations,
                variables=variables,
                include_gridded_metrics=True,
                include_station_metrics=True,
            )

            results["ensemble_bar_plots"].extend(bar_plot_paths)
            print(f"  Created {len(bar_plot_paths)} ensemble metrics bar plots")

        except Exception as e:
            print(f"  Error creating ensemble metrics bar plots: {e}")
            import traceback

            traceback.print_exc()

    # Calculate comprehensive metrics
    if calculate_metrics:
        print(f"\nCalculating comprehensive metrics...")
        try:
            all_metrics = experiment.calculate_comprehensive_metrics(
                stations=stations,
                variables=variables,
                save_to_csv=True,
            )

            csv_files = list(config.output_path.glob("*metrics*.csv"))
            if csv_files:
                results["metrics_files"].extend([str(f) for f in csv_files])
                print(f"  Metrics saved to: {', '.join([f.name for f in csv_files])}")
            else:
                print("  Warning: No metrics CSV files found")

        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            import traceback

            traceback.print_exc()

    # Print summary (updated for ensemble bar plots)
    print(f"\nAnalysis Complete")
    print(f"=================")
    print(f"Timeseries plots: {len(results['timeseries_plots'])}")
    print(f"Scatter plots: {len(results['scatter_plots'])}")
    print(f"Gridded scatter plots: {len(results['gridded_scatter_plots'])}")
    print(f"RMSE maps: {len(results['rmse_maps'])}")
    print(f"Snapshot maps: {len(results['snapshots'])}")
    print(f"Ensemble bar plots: {len(results['ensemble_bar_plots'])}")  # NEW
    print(f"Metrics files: {len(results['metrics_files'])}")
    print(f"Output directory: {config.output_path}")

    # Updated folder structure display
    print(f"\nFolder Structure Created:")
    print(f"├── timeseries/")
    print(f"├── scatter/")
    print(f"│   ├── turboswan_base/")
    print(f"│   │   ├── stations/")
    print(f"│   │   └── stations_aggregate/")
    if create_gridded_scatter:
        print(f"│   ├── turboswan_v0003/")
        print(f"│   │   ├── stations/")
        print(f"│   │   ├── stations_aggregate/")
        print(f"│   │   └── stations_aggregate_ensemble/")
        print(f"│   └── gridded/")
    else:
        print(f"│   └── turboswan_v0003/")
        print(f"│       ├── stations/")
        print(f"│       ├── stations_aggregate/")
        print(f"│       └── stations_aggregate_ensemble/")
    if create_rmse_maps or create_snapshots:
        print(f"├── maps/")
        if create_rmse_maps:
            print(f"│   ├── rmse_maps/")
            for var in variables:
                print(f"│   │   ├── {var.lower()}/")
                print(f"│   │   │   ├── *_base_RMSE_map_*.png")
                print(f"│   │   │   ├── *_v0003_ens*_RMSE_map_*.png")
                print(f"│   │   │   └── *_v0003_ensemble_RMSE_map_*.png")
        if create_snapshots:
            if create_rmse_maps:
                print(f"│   └── snapshots/")
            else:
                print(f"│   └── snapshots/")
            for var in variables:
                print(f"│       └── {var.lower()}/")
                print(f"│           └── *_snapshot_*.png")
    if create_ensemble_bar_plots:
        print(f"├── bar_plots/")
        for var in variables:
            print(f"│   └── {var.lower()}/")
            print(f"│       ├── {var}_rmse_station_averaged_ensemble_bar.png")
            print(f"│       ├── {var}_si_station_averaged_ensemble_bar.png")
            print(f"│       ├── {var}_r_squared_station_averaged_ensemble_bar.png")
            print(f"│       ├── {var}_rel_bias_station_averaged_ensemble_bar.png")
            print(f"│       ├── {var}_rmse_gridded_ensemble_bar.png")
            print(f"│       ├── {var}_si_gridded_ensemble_bar.png")
            print(f"│       ├── {var}_r_squared_gridded_ensemble_bar.png")
            print(f"│       └── {var}_rel_bias_gridded_ensemble_bar.png")
    print(f"└── summary_metrics.csv")

    return results


def v0004_v0016():
    """
    Validate v0004, v0016

    """
    print("Validation v0004, v0016")
    print("==============")

    stations = [
        "North Cormorant 1",
        "A122",
        "K13a",
        "EurogeulE13",
        "L91",
        "wadden eierlandse gat",
        "schiermonnikoog noord",
        "ijmuiden munitiestort 1",
    ]

    map_snapshot_timestamps = [
        "2022-02-18T00:00:00",
    ]

    results = run_validation_analysis(
        experiment_name="v0004_v0016_ensemble_matched",
        time_period=(
            "2022-10-15",
            "2022-11-01",
        ),  # ("2022-01-01", "2022-12-31"), #("2022-01-01", "2022-12-31"), # ("2022-02-15", "2022-03-01") ,("2022-04-20", "2022-05-10"), ("2022-10-15", "2022-11-01"),
        variables=[
            "Hsig"
        ],  # ["Hsig", "Hswell", "Tm_10"], #["Hsig", "Hswell", "Tm_10", "Theta0", "TPsmoo"],
        stations=stations,  # ["A122", "K13a"],  # Single station
        selected_variants=[
            "v0004",
            "v0016",
        ],  # , "v0003", "v0004", "v0005", "v0006", "v0007"],#, "v0003"],
        selected_ensemble_members=[27],  # Single ensemble member ,27,28,29
        # use_ensemble_specific_swan=True,  # Compare each variant against its corresponding SWAN ensemble member
        create_timeseries=True,  # timeseries
        create_scatter=False,  # scatter stations
        create_gridded_scatter=False,  # gridded scatter
        create_rmse_maps=False,  # rmse gridded ensemble
        create_snapshots=False,  # Plot map snapshopts specified
        create_ensemble_bar_plots=False,  # Enable ensemble bar plots
        snapshot_timestamps=map_snapshot_timestamps,
        calculate_metrics=False,  # metrics saved as csv
    )

    return results


def v0017():
    """
    Validate v0017

    """
    print("Validation v0017")
    print("==============")

    stations = [
        "North Cormorant 1",
        "A122",
        "K13a",
        "EurogeulE13",
        "L91",
        "wadden eierlandse gat",
        "schiermonnikoog noord",
        "ijmuiden munitiestort 1",
    ]

    map_snapshot_timestamps = [
        "1994-06-28T00:00:00",
    ]

    results = run_validation_analysis(
        experiment_name="v0017",
        time_period=(
            "1994-01-01",
            "1995-01-01",
        ),  # ("2022-01-01", "2022-12-31"), #("2022-01-01", "2022-12-31"), # ("2022-02-15", "2022-03-01") ,("2022-04-20", "2022-05-10"), ("2022-10-15", "2022-11-01"),
        variables=[
            "Hsig"
        ],  # ["Hsig", "Hswell", "Tm_10"], #["Hsig", "Hswell", "Tm_10", "Theta0", "TPsmoo"],
        stations=stations,  # ["A122", "K13a"],  # Single station
        selected_variants=[
            "v0017"
        ],  # , "v0003", "v0004", "v0005", "v0006", "v0007"],#, "v0003"],
        selected_ensemble_members=[
            1994,
        ],  # Single ensemble member ,27,28,29
        # use_ensemble_specific_swan=True,  # Compare each variant against its corresponding SWAN ensemble member
        create_timeseries=False,  # timeseries
        create_scatter=False,  # scatter stations
        create_gridded_scatter=False,  # gridded scatter
        create_rmse_maps=True,  # rmse gridded ensemble
        create_snapshots=True,  # Plot map snapshopts specified
        create_ensemble_bar_plots=False,  # Enable ensemble bar plots
        snapshot_timestamps=map_snapshot_timestamps,
        calculate_metrics=True,  # metrics saved as csv
    )

    return results


def v0002_v0003_v0004_v0005_v0006_v0007_v0008_v0009_v0010_v0011_v0012_v0014_v0016():
    """
    Validate v0004, v0016

    """
    print("Validation v0004, v0016")
    print("==============")

    stations = [
        "North Cormorant 1",
        "A122",
        "K13a",
        "EurogeulE13",
        "L91",
        "wadden eierlandse gat",
        "schiermonnikoog noord",
        "ijmuiden munitiestort 1",
    ]

    map_snapshot_timestamps = [
        "2022-02-18T00:00:00",
    ]

    results = run_validation_analysis(
        experiment_name="v0002_v0003_v0004_v0005_v0006_v0007_v0008_v0009_v0010_v0011_v0012_v0014_v0016",
        time_period=(
            "2022-01-01",
            "2022-12-31",
        ),  # ("2022-01-01", "2022-12-31"), # ("2022-02-15", "2022-03-01") ,("2022-04-20", "2022-05-10"), ("2022-10-15", "2022-11-01"),
        variables=[
            "Hsig"
        ],  # ["Hsig", "Hswell", "Tm_10"], #["Hsig", "Hswell", "Tm_10", "Theta0", "TPsmoo"],
        stations=stations,  # ["A122", "K13a"],  # Single station
        selected_variants=[
            "v0002",
            "v0003",
            "v0004",
            "v0005",
            "v0006",
            "v0007",
            "v0008",
            "v0009",
            "v0010",
            "v0011",
            "v0012",
            "v0014",
            "v0016",
        ],  # , "v0003", "v0004", "v0005", "v0006", "v0007"],#, "v0003"],
        selected_ensemble_members=[26, 27, 28, 29],  # Single ensemble member ,27,28,29
        # use_ensemble_specific_swan=True,  # Compare each variant against its corresponding SWAN ensemble member
        create_timeseries=False,  # timeseries
        create_scatter=False,  # scatter stations
        create_gridded_scatter=False,  # gridded scatter
        create_rmse_maps=False,  # rmse gridded ensemble
        create_snapshots=False,  # Plot map snapshopts specified
        create_ensemble_bar_plots=True,  # Enable ensemble bar plots
        snapshot_timestamps=map_snapshot_timestamps,
        calculate_metrics=False,  # metrics saved as csv
    )

    return results


def main():
    """
    Main function with different analysis options.
    """
    print("Wave Model Validation (ML TurboSWAN vs. SWAN & Observed)")
    print("===============================")
    print()

    # Validate file structure first
    from turboswanvalid.utils_paths import validate_file_structure

    base_path = "/gpfs/work2/0/prjs1027"

    print("Validating file structure...")
    path_checks = validate_file_structure(base_path)
    missing_paths = [path for path, exists in path_checks.items() if not exists]

    if missing_paths:
        print(f"Warning: Missing paths detected: {missing_paths}")
    else:
        print("All required paths found")

    print("Running validation v0017")
    v0017()
    try:
        quick_results = v0017()
    except Exception as e:
        print(f"Validation v0017 failed: {e}")
        return
    print("Validation v0017 completed successfully")

    # print("running validation v0002_v0003_v0004_v0005_v0006_v0007_v0008_v0009_v0010_v0011_v0012_v0014_v0016")
    # try:
    #     quick_results = v0002_v0003_v0004_v0005_v0006_v0007_v0008_v0009_v0010_v0011_v0012_v0014_v0016()
    # except Exception as e:
    #     print(f"Validation v0002, v0003, v0004 and v0005, v0006,v0007, v0008,v0009,v0010,v0011,v0012,v0014 failed: {e}")


if __name__ == "__main__":
    # Run main analysis
    main()
