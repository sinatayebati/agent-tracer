"""
TRACER Diagnostic and Optimization Script

This script provides comprehensive diagnostics and optimization for TRACER,
including alternative formulations to maximize AUROC performance.

Approach (Option 3 - Hybrid):
1. Diagnostic analysis to understand current failures
2. Implement top 3 most promising variants
3. Grid search optimization for each variant
4. Select best performer

Usage:
    # Run full diagnostic + optimization
    python -m tau2.scripts.diagnose_and_optimize_tracer \\
        data/uncertainty/TIMESTAMP_airline_*.json
    
    # Diagnostic only
    python -m tau2.scripts.diagnose_and_optimize_tracer \\
        data/uncertainty/TIMESTAMP_airline_*.json \\
        --diagnostic-only
    
    # Test specific variant
    python -m tau2.scripts.diagnose_and_optimize_tracer \\
        data/uncertainty/TIMESTAMP_airline_*.json \\
        --variant multiplicative
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tau2.metrics.uncertainty import TRACERConfig, calculate_tracer_score


@dataclass
class DiagnosticResults:
    """Results from diagnostic analysis."""
    
    # Component correlations with failure
    ui_correlation: float
    da_correlation: float
    do_agent_correlation: float
    do_user_correlation: float
    
    # Statistics by outcome
    ui_failures_mean: float
    ui_successes_mean: float
    da_failures_mean: float
    da_successes_mean: float
    
    # Distribution analysis
    failure_rate: float
    num_failures: int
    num_successes: int
    
    # Signal quality metrics
    ui_separation: float  # How well U_i separates failures from successes
    penalty_separation: float  # How well penalties separate


def calculate_point_biserial_correlation(continuous_var: np.ndarray, binary_var: np.ndarray) -> float:
    """
    Calculate point-biserial correlation between a continuous variable and binary outcome.
    
    Args:
        continuous_var: Continuous values (e.g., uncertainty scores)
        binary_var: Binary labels (0/1, where 1=failure)
        
    Returns:
        Correlation coefficient in [-1, 1]
    """
    if len(continuous_var) == 0 or len(np.unique(binary_var)) < 2:
        return 0.0
    
    # Separate into two groups
    group_1 = continuous_var[binary_var == 1]
    group_2 = continuous_var[binary_var == 0]
    
    if len(group_1) == 0 or len(group_2) == 0:
        return 0.0
    
    # Calculate means
    mean_1 = np.mean(group_1)
    mean_2 = np.mean(group_2)
    
    # Calculate standard deviation of all data
    std_all = np.std(continuous_var)
    
    if std_all == 0:
        return 0.0
    
    # Calculate proportions
    n_1 = len(group_1)
    n_2 = len(group_2)
    n = len(continuous_var)
    
    # Point-biserial correlation formula
    correlation = ((mean_1 - mean_2) / std_all) * np.sqrt((n_1 * n_2) / (n * n))
    
    return float(correlation)


def run_diagnostic_analysis(analyzed_sims: list[dict]) -> DiagnosticResults:
    """
    Run comprehensive diagnostic analysis on the data.
    
    Analyzes:
    - Which components (U_i, Da, Do) correlate with failure
    - Distribution differences between failures and successes
    - Signal quality and separability
    
    Args:
        analyzed_sims: List of analyzed simulation dictionaries
        
    Returns:
        DiagnosticResults with all diagnostic metrics
    """
    console = Console()
    console.print("\n[bold cyan]🔬 Running Diagnostic Analysis[/bold cyan]\n")
    
    # Collect all data
    all_ui = []
    all_da = []
    all_do_agent = []
    all_do_user = []
    all_labels = []  # 1=failure, 0=success
    
    for sim in analyzed_sims:
        if sim.get('ground_truth_pass') is None:
            continue
        
        label = 0 if sim['ground_truth_pass'] else 1
        
        # Extract step-level data
        for score in sim['uncertainty_scores']:
            if score['actor'] == 'agent':  # Focus on agent for now
                all_ui.append(score['ui_score'])
                all_da.append(score['da_score'] if score['da_score'] is not None else 0.0)
                all_do_agent.append(score['do_score'] if score['do_type'] == 'agent_coherence' and score['do_score'] is not None else 0.0)
                all_do_user.append(score['do_score'] if score['do_type'] == 'user_coherence' and score['do_score'] is not None else 0.0)
                all_labels.append(label)
    
    all_ui = np.array(all_ui)
    all_da = np.array(all_da)
    all_do_agent = np.array(all_do_agent)
    all_do_user = np.array(all_do_user)
    all_labels = np.array(all_labels)
    
    # Calculate correlations
    ui_corr = calculate_point_biserial_correlation(all_ui, all_labels)
    da_corr = calculate_point_biserial_correlation(all_da, all_labels)
    do_agent_corr = calculate_point_biserial_correlation(all_do_agent, all_labels)
    do_user_corr = calculate_point_biserial_correlation(all_do_user, all_labels)
    
    # Statistics by outcome
    failures = all_labels == 1
    successes = all_labels == 0
    
    ui_failures_mean = float(np.mean(all_ui[failures])) if np.any(failures) else 0.0
    ui_successes_mean = float(np.mean(all_ui[successes])) if np.any(successes) else 0.0
    da_failures_mean = float(np.mean(all_da[failures])) if np.any(failures) else 0.0
    da_successes_mean = float(np.mean(all_da[successes])) if np.any(successes) else 0.0
    
    # Failure statistics
    num_failures = int(np.sum(all_labels))
    num_successes = len(all_labels) - num_failures
    failure_rate = num_failures / len(all_labels) if len(all_labels) > 0 else 0.0
    
    # Signal separation (effect size)
    ui_separation = abs(ui_failures_mean - ui_successes_mean) / (np.std(all_ui) + 1e-10)
    penalty_mean_failures = (da_failures_mean + float(np.mean(all_do_agent[failures])) + float(np.mean(all_do_user[failures]))) / 3
    penalty_mean_successes = (da_successes_mean + float(np.mean(all_do_agent[successes])) + float(np.mean(all_do_user[successes]))) / 3
    penalty_separation = abs(penalty_mean_failures - penalty_mean_successes) / (np.std(all_da) + 1e-10)
    
    results = DiagnosticResults(
        ui_correlation=ui_corr,
        da_correlation=da_corr,
        do_agent_correlation=do_agent_corr,
        do_user_correlation=do_user_corr,
        ui_failures_mean=ui_failures_mean,
        ui_successes_mean=ui_successes_mean,
        da_failures_mean=da_failures_mean,
        da_successes_mean=da_successes_mean,
        failure_rate=failure_rate,
        num_failures=num_failures,
        num_successes=num_successes,
        ui_separation=ui_separation,
        penalty_separation=penalty_separation
    )
    
    # Display results
    console.print("[bold]Component Correlations with Failure:[/bold]")
    console.print(f"  U_i (Normalized Entropy):  {ui_corr:+.4f}")
    console.print(f"  Da (Inquiry Drift):        {da_corr:+.4f}")
    console.print(f"  Do_agent (Agent Coherence): {do_agent_corr:+.4f}")
    console.print(f"  Do_user (User Coherence):   {do_user_corr:+.4f}")
    
    console.print(f"\n[bold]Signal Quality:[/bold]")
    console.print(f"  U_i separation (effect size):     {ui_separation:.4f}")
    console.print(f"  Penalty separation (effect size): {penalty_separation:.4f}")
    
    console.print(f"\n[bold]Distribution Analysis:[/bold]")
    console.print(f"  Failure rate: {failure_rate:.1%} ({num_failures}/{num_failures+num_successes})")
    console.print(f"  U_i (failures):  mean={ui_failures_mean:.4f}")
    console.print(f"  U_i (successes): mean={ui_successes_mean:.4f}")
    console.print(f"  Da (failures):   mean={da_failures_mean:.4f}")
    console.print(f"  Da (successes):  mean={da_successes_mean:.4f}")
    
    # Interpretation
    console.print(f"\n[bold yellow]💡 Key Insights:[/bold yellow]")
    
    if abs(ui_corr) > 0.1:
        console.print(f"  ✓ U_i shows {'positive' if ui_corr > 0 else 'negative'} correlation with failure")
    else:
        console.print(f"  ⚠️  U_i shows weak correlation - may not be predictive on its own")
    
    if abs(da_corr) > 0.05:
        console.print(f"  ✓ Da (repetition) shows signal")
    else:
        console.print(f"  ⚠️  Da shows weak signal - may be adding noise")
    
    if abs(do_agent_corr) > 0.05 or abs(do_user_corr) > 0.05:
        console.print(f"  ✓ Do (coherence) shows signal")
    else:
        console.print(f"  ⚠️  Do shows weak signal - may be adding noise")
    
    if ui_separation < 0.2:
        console.print(f"  [red]⚠️  Low U_i separation - failures and successes have similar uncertainty![/red]")
    
    return results


def calculate_variant_score(
    step_data: list[dict],
    variant: str,
    config: TRACERConfig
) -> float:
    """
    Calculate score using different TRACER variants.
    
    Args:
        step_data: List of step dictionaries with ui, da, do_agent, do_user
        variant: Which variant to use ('additive', 'multiplicative', 'max', 'separate')
        config: TRACER configuration
        
    Returns:
        Aggregate score for the trajectory
    """
    if not step_data:
        return 0.0
    
    if variant == 'additive':
        # Original TRACER: risk_i = U_i + penalty
        step_risks = []
        for step in step_data:
            ui = step.get('ui', 0.0)
            da = step.get('da', 0.0) if step.get('da') is not None else 0.0
            do_agent = step.get('do_agent', 0.0) if step.get('do_agent') is not None else 0.0
            do_user = step.get('do_user', 0.0) if step.get('do_user') is not None else 0.0
            
            penalty = config.alpha * da + config.beta * do_agent + config.gamma * do_user
            risk = ui + penalty
            step_risks.append(risk)
        
        # Top-k aggregation with ensemble
        N = len(step_risks)
        if config.top_k_percentile >= 1.0:
            top_k_risks = step_risks
        else:
            sorted_risks = sorted(step_risks, reverse=True)
            top_k_count = max(1, int(config.top_k_percentile * N))
            top_k_risks = sorted_risks[:top_k_count]
        
        mean_top_k = float(np.mean(top_k_risks))
        
        if config.ensemble_weight_max > 0.0:
            max_risk = float(np.max(step_risks))
            score = (1 - config.ensemble_weight_max) * mean_top_k + config.ensemble_weight_max * max_risk
        else:
            score = mean_top_k
        
        return score
    
    elif variant == 'multiplicative':
        # Variant A: risk_i = U_i × (1 + penalty)
        step_risks = []
        for step in step_data:
            ui = step.get('ui', 0.0)
            da = step.get('da', 0.0) if step.get('da') is not None else 0.0
            do_agent = step.get('do_agent', 0.0) if step.get('do_agent') is not None else 0.0
            do_user = step.get('do_user', 0.0) if step.get('do_user') is not None else 0.0
            
            penalty = config.alpha * da + config.beta * do_agent + config.gamma * do_user
            weight = 1.0 + penalty
            risk = ui * weight
            step_risks.append(risk)
        
        # Same aggregation as additive
        N = len(step_risks)
        if config.top_k_percentile >= 1.0:
            top_k_risks = step_risks
        else:
            sorted_risks = sorted(step_risks, reverse=True)
            top_k_count = max(1, int(config.top_k_percentile * N))
            top_k_risks = sorted_risks[:top_k_count]
        
        mean_top_k = float(np.mean(top_k_risks))
        
        if config.ensemble_weight_max > 0.0:
            max_risk = float(np.max(step_risks))
            score = (1 - config.ensemble_weight_max) * mean_top_k + config.ensemble_weight_max * max_risk
        else:
            score = mean_top_k
        
        return score
    
    elif variant == 'max':
        # Variant B: risk_i = max(U_i, α·Da, β·Do_agent, γ·Do_user)
        step_risks = []
        for step in step_data:
            ui = step.get('ui', 0.0)
            da = step.get('da', 0.0) if step.get('da') is not None else 0.0
            do_agent = step.get('do_agent', 0.0) if step.get('do_agent') is not None else 0.0
            do_user = step.get('do_user', 0.0) if step.get('do_user') is not None else 0.0
            
            risk = max(ui, config.alpha * da, config.beta * do_agent, config.gamma * do_user)
            step_risks.append(risk)
        
        # Same aggregation
        N = len(step_risks)
        if config.top_k_percentile >= 1.0:
            top_k_risks = step_risks
        else:
            sorted_risks = sorted(step_risks, reverse=True)
            top_k_count = max(1, int(config.top_k_percentile * N))
            top_k_risks = sorted_risks[:top_k_count]
        
        mean_top_k = float(np.mean(top_k_risks))
        
        if config.ensemble_weight_max > 0.0:
            max_risk = float(np.max(step_risks))
            score = (1 - config.ensemble_weight_max) * mean_top_k + config.ensemble_weight_max * max_risk
        else:
            score = mean_top_k
        
        return score
    
    elif variant == 'separate':
        # Variant C: Separate channels
        # uncertainty_trajectory = mean(U_i)
        # penalty_trajectory = mean(penalties)
        # TRACER = w1·uncertainty + w2·penalty
        
        ui_values = []
        penalties = []
        
        for step in step_data:
            ui = step.get('ui', 0.0)
            da = step.get('da', 0.0) if step.get('da') is not None else 0.0
            do_agent = step.get('do_agent', 0.0) if step.get('do_agent') is not None else 0.0
            do_user = step.get('do_user', 0.0) if step.get('do_user') is not None else 0.0
            
            ui_values.append(ui)
            penalty = config.alpha * da + config.beta * do_agent + config.gamma * do_user
            penalties.append(penalty)
        
        # Use top-k for both channels
        N = len(ui_values)
        if config.top_k_percentile >= 1.0:
            uncertainty_signal = float(np.mean(ui_values))
            penalty_signal = float(np.mean(penalties))
        else:
            sorted_ui = sorted(ui_values, reverse=True)
            sorted_penalties = sorted(penalties, reverse=True)
            top_k_count = max(1, int(config.top_k_percentile * N))
            uncertainty_signal = float(np.mean(sorted_ui[:top_k_count]))
            penalty_signal = float(np.mean(sorted_penalties[:top_k_count]))
        
        # Combine with weights (use ensemble_weight_max as weight for penalties)
        w1 = 1.0 - config.ensemble_weight_max
        w2 = config.ensemble_weight_max
        score = w1 * uncertainty_signal + w2 * penalty_signal
        
        return score
    
    else:
        raise ValueError(f"Unknown variant: {variant}")


def evaluate_variant_auroc(
    analyzed_sims: list[dict],
    variant: str,
    config: TRACERConfig
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate a TRACER variant and return its AUROC.
    
    Args:
        analyzed_sims: List of analyzed simulations
        variant: Which variant to use
        config: TRACER configuration
        
    Returns:
        Tuple of (auroc, y_true, y_scores)
    """
    y_scores = []
    y_true = []
    
    for sim in analyzed_sims:
        if sim.get('ground_truth_pass') is None:
            continue
        
        # Build step_data
        step_data = []
        for score in sim['uncertainty_scores']:
            step_data.append({
                'ui': score['ui_score'],
                'da': score['da_score'],
                'do_agent': score['do_score'] if score.get('do_type') == 'agent_coherence' else None,
                'do_user': score['do_score'] if score.get('do_type') == 'user_coherence' else None
            })
        
        if not step_data:
            continue
        
        # Calculate score using variant
        score = calculate_variant_score(step_data, variant, config)
        
        # Label encoding: Failure=1, Success=0
        ground_truth = 0 if sim['ground_truth_pass'] else 1
        
        y_scores.append(score)
        y_true.append(ground_truth)
    
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    if len(y_scores) < 2 or len(np.unique(y_true)) < 2:
        return 0.0, y_true, y_scores
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
        return float(auroc), y_true, y_scores
    except Exception as e:
        logger.warning(f"Failed to calculate AUROC: {e}")
        return 0.0, y_true, y_scores


def grid_search_variant(
    analyzed_sims: list[dict],
    variant: str,
    console: Console
) -> tuple[TRACERConfig, float]:
    """
    Run grid search to find optimal parameters for a variant.
    
    Args:
        analyzed_sims: List of analyzed simulations
        variant: Which variant to optimize
        console: Rich console for output
        
    Returns:
        Tuple of (best_config, best_auroc)
    """
    console.print(f"\n[bold cyan]Optimizing {variant.upper()} variant...[/bold cyan]")
    
    # Define search grid
    alpha_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    beta_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    gamma_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    topk_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0]
    ensemble_values = [0.0, 0.1, 0.2, 0.3]
    
    best_auroc = 0.0
    best_config = None
    total_configs = 0
    
    # Progress tracking
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        total_iterations = len(alpha_values) * len(beta_values) * len(gamma_values) * len(topk_values) * len(ensemble_values)
        task = progress.add_task(f"Testing {variant}...", total=total_iterations)
        
        for alpha in alpha_values:
            for beta in beta_values:
                for gamma in gamma_values:
                    for topk in topk_values:
                        for ensemble in ensemble_values:
                            config = TRACERConfig(
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                top_k_percentile=topk,
                                ensemble_weight_max=ensemble
                            )
                            
                            auroc, _, _ = evaluate_variant_auroc(analyzed_sims, variant, config)
                            total_configs += 1
                            
                            if auroc > best_auroc:
                                best_auroc = auroc
                                best_config = config
                            
                            progress.update(task, advance=1)
    
    console.print(f"  Tested {total_configs} configurations")
    console.print(f"  Best AUROC: [bold green]{best_auroc:.4f}[/bold green]")
    console.print(f"  Parameters: α={best_config.alpha}, β={best_config.beta}, γ={best_config.gamma}, " +
                  f"top-k={best_config.top_k_percentile:.2f}, ensemble={best_config.ensemble_weight_max:.2f}")
    
    return best_config, best_auroc


def main():
    parser = argparse.ArgumentParser(description="Diagnostic and Optimization for TRACER")
    parser.add_argument("input_file", type=str, help="Path to uncertainty analysis JSON file")
    parser.add_argument("--diagnostic-only", action="store_true", help="Run diagnostic analysis only")
    parser.add_argument("--variant", type=str, choices=['additive', 'multiplicative', 'max', 'separate', 'all'],
                        default='all', help="Which variant to test")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    console = Console()
    
    # Load data
    console.print(f"\n[bold]Loading data from:[/bold] {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    analyzed_sims = data['results']
    console.print(f"Loaded {len(analyzed_sims)} simulations")
    
    # Run diagnostic analysis
    diagnostic_results = run_diagnostic_analysis(analyzed_sims)
    
    if args.diagnostic_only:
        console.print("\n[bold green]✓ Diagnostic analysis complete![/bold green]")
        return
    
    # Test variants
    console.print("\n[bold cyan]🚀 Testing TRACER Variants[/bold cyan]")
    
    variants_to_test = ['additive', 'multiplicative', 'max', 'separate'] if args.variant == 'all' else [args.variant]
    
    results = {}
    for variant in variants_to_test:
        best_config, best_auroc = grid_search_variant(analyzed_sims, variant, console)
        results[variant] = {
            'auroc': best_auroc,
            'config': {
                'alpha': best_config.alpha,
                'beta': best_config.beta,
                'gamma': best_config.gamma,
                'top_k_percentile': best_config.top_k_percentile,
                'ensemble_weight_max': best_config.ensemble_weight_max
            }
        }
    
    # Display comparison
    console.print("\n[bold cyan]📊 Variant Comparison[/bold cyan]\n")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Variant", style="white", width=20)
    table.add_column("AUROC", justify="right", width=10)
    table.add_column("α", justify="right", width=8)
    table.add_column("β", justify="right", width=8)
    table.add_column("γ", justify="right", width=8)
    table.add_column("Top-K", justify="right", width=10)
    table.add_column("Ensemble", justify="right", width=10)
    
    best_variant = max(results.items(), key=lambda x: x[1]['auroc'])
    
    for variant, result in sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True):
        config = result['config']
        is_best = variant == best_variant[0]
        
        table.add_row(
            f"[bold]{variant.upper()}[/bold]" if is_best else variant.upper(),
            f"[bold green]{result['auroc']:.4f}[/bold green]" if is_best else f"{result['auroc']:.4f}",
            f"{config['alpha']:.1f}",
            f"{config['beta']:.1f}",
            f"{config['gamma']:.1f}",
            f"{config['top_k_percentile']:.2f}",
            f"{config['ensemble_weight_max']:.2f}"
        )
    
    console.print(table)
    
    console.print(f"\n[bold green]✓ Best variant: {best_variant[0].upper()} (AUROC={best_variant[1]['auroc']:.4f})[/bold green]")
    
    # Save results
    if not args.no_save:
        output_path = args.output if args.output else Path(args.input_file).parent / f"diagnostic_{Path(args.input_file).name}"
        
        output_data = {
            'diagnostic_results': {
                'ui_correlation': diagnostic_results.ui_correlation,
                'da_correlation': diagnostic_results.da_correlation,
                'do_agent_correlation': diagnostic_results.do_agent_correlation,
                'do_user_correlation': diagnostic_results.do_user_correlation,
                'ui_separation': diagnostic_results.ui_separation,
                'penalty_separation': diagnostic_results.penalty_separation,
                'failure_rate': diagnostic_results.failure_rate
            },
            'optimization_results': results,
            'best_variant': best_variant[0],
            'best_auroc': best_variant[1]['auroc'],
            'best_config': best_variant[1]['config']
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    main()
