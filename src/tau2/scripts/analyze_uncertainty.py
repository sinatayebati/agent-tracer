"""
Uncertainty Analysis Script

Analyzes Tau-2 simulation trajectories and calculates uncertainty metrics
using the SAUP framework.

By default, results are saved to data/uncertainty/ with the same filename as the input.

Usage:
    # Basic analysis (auto-saves to data/uncertainty/)
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json
    
    # Detailed turn-by-turn view
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json --detailed
    
    # Custom output location
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json --output results/my_analysis.json
    
    # Don't save, just display
    python -m tau2.scripts.analyze_uncertainty data/simulations/sim_file.json --no-save

Example:
    Input:  data/simulations/2025-11-06_airline_gemini.json
    Output: data/uncertainty/2025-11-06_airline_gemini.json (auto-saved)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from tau2.data_model.simulation import Results
from tau2.metrics.uncertainty import (
    SAUPConfig,
    calculate_normalized_entropy,
    calculate_saup_score,
    get_uncertainty_stats,
)

# Optional sklearn import for AUROC calculation
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. AUROC evaluation will be skipped.")


class TurnUncertainty(BaseModel):
    """Uncertainty information for a single turn."""

    turn: int
    actor: str
    role: str
    ui_score: float
    content_preview: str
    statistics: Optional[dict] = None
    da_score: Optional[float] = None
    do_score: Optional[float] = None
    do_type: Optional[str] = None


class SimulationUncertainty(BaseModel):
    """Uncertainty analysis for a single simulation."""

    simulation_id: str
    task_id: str
    trial: int
    turn_count: int
    uncertainty_scores: list[TurnUncertainty]
    summary: dict
    saup_metrics: Optional[dict] = None
    ground_truth_pass: Optional[bool] = None


class AUROCMetrics(BaseModel):
    """AUROC evaluation metrics for failure prediction."""
    
    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    optimal_threshold: float
    num_samples: int
    num_failures: int
    num_successes: int
    mean_saup_failures: float
    mean_saup_successes: float
    std_saup_failures: float
    std_saup_successes: float


class UncertaintyAnalysis(BaseModel):
    """Complete uncertainty analysis results."""

    metadata: dict
    results: list[SimulationUncertainty]
    auroc_metrics: Optional[AUROCMetrics] = None


def analyze_simulation(simulation: dict, verbose: bool = False) -> SimulationUncertainty:
    """
    Analyze a single simulation and extract uncertainty scores.

    Args:
        simulation: A simulation run dictionary
        verbose: If True, include detailed statistics

    Returns:
        SimulationUncertainty: Analyzed uncertainty data
    """
    uncertainty_scores = []
    turn_counter = 0

    for message in simulation.get("messages", []):
        role = message.get("role")

        # Only process agent and user messages
        if role not in ["assistant", "user"]:
            continue

        turn_counter += 1

        # Determine actor type
        actor = "agent" if role == "assistant" else "user"

        # Extract logprobs and calculate uncertainty
        logprobs = message.get("logprobs")
        ui_score = calculate_normalized_entropy(logprobs)
        
        # Extract semantic distance metrics
        da_score = message.get("da_score")
        do_score = message.get("do_score")
        do_type = message.get("do_type")

        # Build turn data
        turn_data = TurnUncertainty(
            turn=turn_counter,
            actor=actor,
            role=role,
            ui_score=ui_score,
            content_preview=(
                message.get("content", "")[:100]
                if message.get("content")
                else "[tool_call]"
            ),
            da_score=da_score,
            do_score=do_score,
            do_type=do_type,
        )

        # Add detailed statistics if requested
        if verbose and logprobs is not None:
            stats = get_uncertainty_stats(logprobs)
            turn_data.statistics = stats.model_dump()

        uncertainty_scores.append(turn_data)

    # Calculate SAUP-D aggregation score
    saup_metrics = None
    if uncertainty_scores:
        step_data = [
            {
                "ui": turn.ui_score,
                "da": turn.da_score,
                "do_agent": turn.do_score if turn.do_type == "agent_coherence" else None,
                "do_user": turn.do_score if turn.do_type == "user_coherence" else None
            }
            for turn in uncertainty_scores
        ]
        saup_result = calculate_saup_score(step_data, SAUPConfig())
        # Remove weights list (too verbose)
        saup_metrics = {k: v for k, v in saup_result.items() if k != 'weights'}
    
    # Extract ground truth (task success)
    ground_truth = simulation.get("reward_info", {}).get("reward", None) if simulation.get("reward_info") else None
    ground_truth_pass = ground_truth == 1.0 if ground_truth is not None else None

    # Calculate summary statistics
    summary = {}
    if uncertainty_scores:
        agent_scores = [s.ui_score for s in uncertainty_scores if s.actor == "agent"]
        user_scores = [s.ui_score for s in uncertainty_scores if s.actor == "user"]
        
        # Semantic distance metrics
        da_scores = [s.da_score for s in uncertainty_scores if s.da_score is not None]
        do_scores = [s.do_score for s in uncertainty_scores if s.do_score is not None]
        do_agent_coherence = [s.do_score for s in uncertainty_scores if s.do_score is not None and s.do_type == "agent_coherence"]
        do_user_coherence = [s.do_score for s in uncertainty_scores if s.do_score is not None and s.do_type == "user_coherence"]

        summary = {
            "mean_uncertainty_overall": float(
                np.mean([s.ui_score for s in uncertainty_scores])
            ),
            "mean_uncertainty_agent": float(np.mean(agent_scores)) if agent_scores else 0.0,
            "mean_uncertainty_user": float(np.mean(user_scores)) if user_scores else 0.0,
            "max_uncertainty_overall": float(
                np.max([s.ui_score for s in uncertainty_scores])
            ),
            "agent_turn_count": len(agent_scores),
            "user_turn_count": len(user_scores),
            # Semantic distance metrics
            "mean_da_score": float(np.mean(da_scores)) if da_scores else None,
            "std_da_score": float(np.std(da_scores)) if da_scores else None,
            "mean_do_score": float(np.mean(do_scores)) if do_scores else None,
            "std_do_score": float(np.std(do_scores)) if do_scores else None,
            "mean_do_agent_coherence": float(np.mean(do_agent_coherence)) if do_agent_coherence else None,
            "mean_do_user_coherence": float(np.mean(do_user_coherence)) if do_user_coherence else None,
            "da_count": len(da_scores),
            "do_count": len(do_scores),
        }

    return SimulationUncertainty(
        simulation_id=simulation.get("id", "unknown"),
        task_id=simulation.get("task_id", "unknown"),
        trial=simulation.get("trial", 0),
        turn_count=len(uncertainty_scores),
        uncertainty_scores=uncertainty_scores,
        summary=summary,
        saup_metrics=saup_metrics,
        ground_truth_pass=ground_truth_pass,
    )


def calculate_auroc_metrics(analyzed_sims: list[SimulationUncertainty]) -> Optional[AUROCMetrics]:
    """
    Calculate AUROC metrics for SAUP-D failure prediction.
    
    Hypothesis: High SAUP-D score predicts task failure.
    Label encoding: Failure=1, Success=0
    
    Args:
        analyzed_sims: List of analyzed simulations with SAUP metrics
        
    Returns:
        AUROCMetrics if calculation successful, None otherwise
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available. Skipping AUROC calculation.")
        return None
    
    # Extract SAUP scores and ground truth labels
    y_scores = []
    y_true = []
    
    for sim in analyzed_sims:
        # Need both SAUP score and ground truth
        if sim.saup_metrics is None or sim.ground_truth_pass is None:
            continue
        
        saup_score = sim.saup_metrics.get('saup_score')
        if saup_score is None:
            continue
        
        # Label encoding: Failure=1, Success=0
        ground_truth = 0 if sim.ground_truth_pass else 1
        
        y_scores.append(saup_score)
        y_true.append(ground_truth)
    
    # Need at least 2 samples and both classes present
    if len(y_scores) < 2:
        logger.warning(f"Not enough samples for AUROC ({len(y_scores)} < 2). Skipping.")
        return None
    
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class present in data: {unique_labels}. Cannot calculate AUROC.")
        return None
    
    try:
        # Calculate AUROC
        auroc = roc_auc_score(y_true, y_scores)
        
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred = (y_scores >= optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Statistics by class
        failures = y_scores[y_true == 1]
        successes = y_scores[y_true == 0]
        
        return AUROCMetrics(
            auroc=float(auroc),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            optimal_threshold=float(optimal_threshold),
            num_samples=len(y_scores),
            num_failures=int(np.sum(y_true == 1)),
            num_successes=int(np.sum(y_true == 0)),
            mean_saup_failures=float(np.mean(failures)) if len(failures) > 0 else 0.0,
            mean_saup_successes=float(np.mean(successes)) if len(successes) > 0 else 0.0,
            std_saup_failures=float(np.std(failures)) if len(failures) > 0 else 0.0,
            std_saup_successes=float(np.std(successes)) if len(successes) > 0 else 0.0,
        )
    except Exception as e:
        logger.error(f"Failed to calculate AUROC: {e}")
        return None


def analyze_results(results: Results, verbose: bool = False, calculate_auroc: bool = True) -> UncertaintyAnalysis:
    """
    Analyze all simulations in the results.

    Args:
        results: Tau-2 simulation results
        verbose: If True, include detailed statistics
        calculate_auroc: If True, calculate AUROC metrics for failure prediction

    Returns:
        UncertaintyAnalysis: Complete analysis results
    """
    analyzed_sims = []

    for simulation in results.simulations:
        sim_dict = simulation.model_dump()
        analyzed = analyze_simulation(sim_dict, verbose=verbose)
        analyzed_sims.append(analyzed)

    # Create metadata
    metadata = {
        "source_timestamp": results.timestamp or "unknown",
        "total_simulations": len(analyzed_sims),
        "domain": results.info.environment_info.domain_name,
        "agent_llm": results.info.agent_info.llm,
        "user_llm": results.info.user_info.llm,
    }
    
    # Calculate AUROC metrics if requested and SAUP scores are available
    auroc_metrics = None
    if calculate_auroc:
        auroc_metrics = calculate_auroc_metrics(analyzed_sims)

    return UncertaintyAnalysis(
        metadata=metadata,
        results=analyzed_sims,
        auroc_metrics=auroc_metrics
    )


def print_uncertainty_summary_from_results(
    results: Results, console: Optional[Console] = None
) -> None:
    """
    Print uncertainty summary directly from simulation results.
    
    This is a lightweight version that extracts and displays uncertainty
    statistics from simulation results that have embedded uncertainty data
    (i.e., simulations run with --calculate-uncertainty).
    
    Args:
        results: Simulation results with embedded uncertainty data
        console: Rich console (creates one if not provided)
    """
    if console is None:
        console = Console()
    
    # Collect all uncertainty and semantic distance scores
    all_agent_uncertainties = []
    all_user_uncertainties = []
    all_da_scores = []
    all_do_scores = []
    all_do_agent = []
    all_do_user = []
    sims_with_uncertainty = 0
    
    for sim in results.simulations:
        has_uncertainty = False
        for msg in sim.messages:
            # Only process messages that have the uncertainty attribute (AssistantMessage, UserMessage)
            if hasattr(msg, 'uncertainty') and msg.uncertainty is not None:
                has_uncertainty = True
                uncertainty_score = msg.uncertainty.get('normalized_entropy', 0.0)
                if msg.role == "assistant":
                    all_agent_uncertainties.append(uncertainty_score)
                elif msg.role == "user":
                    all_user_uncertainties.append(uncertainty_score)
            
            # Collect semantic distance metrics
            if hasattr(msg, 'da_score') and msg.da_score is not None:
                all_da_scores.append(msg.da_score)
            if hasattr(msg, 'do_score') and msg.do_score is not None:
                all_do_scores.append(msg.do_score)
                if hasattr(msg, 'do_type'):
                    if msg.do_type == "agent_coherence":
                        all_do_agent.append(msg.do_score)
                    elif msg.do_type == "user_coherence":
                        all_do_user.append(msg.do_score)
        if has_uncertainty:
            sims_with_uncertainty += 1
    
    # If no uncertainty data found, return early
    if not all_agent_uncertainties and not all_user_uncertainties:
        return
    
    # Print summary
    console.print("\n" + "=" * 80, style="dim")
    console.print(
        "UNCERTAINTY ANALYSIS SUMMARY", style="bold cyan", justify="center"
    )
    console.print("=" * 80 + "\n", style="dim")
    
    # Metadata
    console.print(f"[bold]Domain:[/bold] {results.info.environment_info.domain_name}")
    console.print(f"[bold]Agent Model:[/bold] {results.info.agent_info.llm}")
    console.print(f"[bold]User Model:[/bold] {results.info.user_info.llm}")
    console.print(f"[bold]Simulations with Uncertainty:[/bold] {sims_with_uncertainty}/{len(results.simulations)}\n")
    
    # Overall statistics
    console.print("[bold cyan]Overall Statistics[/bold cyan]")
    
    if all_agent_uncertainties:
        console.print("\n[bold]Agent Reasoning Uncertainty (U_i,agent):[/bold]")
        console.print(f"  Mean: {np.mean(all_agent_uncertainties):.4f}")
        console.print(f"  Std:  {np.std(all_agent_uncertainties):.4f}")
        console.print(f"  Min:  {np.min(all_agent_uncertainties):.4f}")
        console.print(f"  Max:  {np.max(all_agent_uncertainties):.4f}")
        console.print(f"  Total turns: {len(all_agent_uncertainties)}")
    
    if all_user_uncertainties:
        console.print("\n[bold]User Confusion (U_i,user):[/bold]")
        console.print(f"  Mean: {np.mean(all_user_uncertainties):.4f}")
        console.print(f"  Std:  {np.std(all_user_uncertainties):.4f}")
        console.print(f"  Min:  {np.min(all_user_uncertainties):.4f}")
        console.print(f"  Max:  {np.max(all_user_uncertainties):.4f}")
        console.print(f"  Total turns: {len(all_user_uncertainties)}")
    
    # Semantic distance metrics
    if all_da_scores or all_do_scores:
        console.print("\n[bold cyan]Semantic Distance Metrics[/bold cyan]")
        
        if all_da_scores:
            console.print("\n[bold]Inquiry Drift (Da):[/bold]")
            console.print(f"  Mean: {np.mean(all_da_scores):.4f}")
            console.print(f"  Std:  {np.std(all_da_scores):.4f}")
            console.print(f"  Min:  {np.min(all_da_scores):.4f}")
            console.print(f"  Max:  {np.max(all_da_scores):.4f}")
            console.print(f"  Total measurements: {len(all_da_scores)}")
        
        if all_do_scores:
            console.print("\n[bold]Inference Gap (Do):[/bold]")
            console.print(f"  Mean: {np.mean(all_do_scores):.4f}")
            console.print(f"  Std:  {np.std(all_do_scores):.4f}")
            console.print(f"  Min:  {np.min(all_do_scores):.4f}")
            console.print(f"  Max:  {np.max(all_do_scores):.4f}")
            console.print(f"  Total measurements: {len(all_do_scores)}")
            
            if all_do_agent:
                console.print(f"\n  [dim]Agent Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_agent):.4f}")
                console.print(f"    Count: {len(all_do_agent)}")
            
            if all_do_user:
                console.print(f"\n  [dim]User Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_user):.4f}")
                console.print(f"    Count: {len(all_do_user)}")
    
    # SAUP-D Aggregation Scores
    all_saup_scores = []
    saup_passed = []
    saup_failed = []
    
    for sim in results.simulations:
        if hasattr(sim, 'saup_metrics') and sim.saup_metrics is not None:
            saup_score = sim.saup_metrics.get('saup_score')
            if saup_score is not None:
                all_saup_scores.append(saup_score)
                
                # Track by ground truth
                if hasattr(sim, 'reward_info') and sim.reward_info is not None:
                    reward = sim.reward_info.reward
                    if reward == 1.0:
                        saup_passed.append(saup_score)
                    else:
                        saup_failed.append(saup_score)
    
    if all_saup_scores:
        console.print("\n[bold cyan]SAUP-D Aggregation Scores[/bold cyan]")
        console.print(f"  Mean SAUP: {np.mean(all_saup_scores):.4f}")
        console.print(f"  Std SAUP:  {np.std(all_saup_scores):.4f}")
        console.print(f"  Min SAUP:  {np.min(all_saup_scores):.4f}")
        console.print(f"  Max SAUP:  {np.max(all_saup_scores):.4f}")
        console.print(f"  Total simulations: {len(all_saup_scores)}")
        
        # Ground truth correlation
        if saup_passed or saup_failed:
            total_with_gt = len(saup_passed) + len(saup_failed)
            console.print(f"\n  Ground Truth: {len(saup_passed)}/{total_with_gt} passed ({100*len(saup_passed)/total_with_gt:.1f}%)")
            if saup_passed:
                console.print(f"  Mean SAUP (passed): {np.mean(saup_passed):.4f}")
            if saup_failed:
                console.print(f"  Mean SAUP (failed): {np.mean(saup_failed):.4f}")
            
            # Quick AUROC estimate (inline calculation for real-time summary)
            if SKLEARN_AVAILABLE and len(saup_passed) > 0 and len(saup_failed) > 0:
                try:
                    y_scores_inline = np.array(saup_passed + saup_failed)
                    y_true_inline = np.array([0]*len(saup_passed) + [1]*len(saup_failed))
                    auroc_inline = roc_auc_score(y_true_inline, y_scores_inline)
                    
                    auroc_color = "green" if auroc_inline > 0.7 else "yellow" if auroc_inline > 0.6 else "red"
                    console.print(f"\n  Quick AUROC: [{auroc_color}]{auroc_inline:.4f}[/{auroc_color}]")
                except Exception:
                    pass  # Silently skip if calculation fails
    
    # Per-simulation summary (first 3)
    console.print("\n[bold cyan]Per-Simulation Summary[/bold cyan] (showing first 3)\n")
    for i, sim in enumerate(results.simulations[:3]):
        agent_scores = []
        user_scores = []
        for msg in sim.messages:
            # Only process messages that have the uncertainty attribute
            if hasattr(msg, 'uncertainty') and msg.uncertainty is not None:
                uncertainty_score = msg.uncertainty.get('normalized_entropy', 0.0)
                if msg.role == "assistant":
                    agent_scores.append(uncertainty_score)
                elif msg.role == "user":
                    user_scores.append(uncertainty_score)
        
        if agent_scores or user_scores:
            console.print(f"[bold]Simulation {i+1}[/bold] (Task: {sim.task_id})")
            console.print(f"  Turns: {len(agent_scores) + len(user_scores)}")
            if agent_scores:
                console.print(f"  Mean uncertainty (agent): {np.mean(agent_scores):.4f}")
            if user_scores:
                console.print(f"  Mean uncertainty (user):  {np.mean(user_scores):.4f}")
            console.print()
    
    if len(results.simulations) > 3:
        console.print(f"... and {len(results.simulations) - 3} more simulations\n")
    
    console.print("=" * 80, style="dim")
    console.print()


def print_summary(analysis: UncertaintyAnalysis, console: Console):
    """Print a summary of the analysis results."""
    console.print("\n" + "=" * 80, style="dim")
    console.print(
        "UNCERTAINTY ANALYSIS SUMMARY", style="bold cyan", justify="center"
    )
    console.print("=" * 80 + "\n", style="dim")

    # Metadata
    metadata = analysis.metadata
    console.print(f"[bold]Domain:[/bold] {metadata.get('domain', 'unknown')}")
    console.print(
        f"[bold]Agent Model:[/bold] {metadata.get('agent_llm', 'unknown')}"
    )
    console.print(f"[bold]User Model:[/bold] {metadata.get('user_llm', 'unknown')}")
    console.print(
        f"[bold]Total Simulations:[/bold] {metadata.get('total_simulations', 0)}\n"
    )

    # Overall statistics
    all_agent_uncertainties = []
    all_user_uncertainties = []

    for sim in analysis.results:
        for score in sim.uncertainty_scores:
            if score.actor == "agent":
                all_agent_uncertainties.append(score.ui_score)
            else:
                all_user_uncertainties.append(score.ui_score)

    if all_agent_uncertainties or all_user_uncertainties:
        console.print("[bold cyan]Overall Statistics[/bold cyan]")

        if all_agent_uncertainties:
            console.print("\n[bold]Agent Reasoning Uncertainty (U_i,agent):[/bold]")
            console.print(f"  Mean: {np.mean(all_agent_uncertainties):.4f}")
            console.print(f"  Std:  {np.std(all_agent_uncertainties):.4f}")
            console.print(f"  Min:  {np.min(all_agent_uncertainties):.4f}")
            console.print(f"  Max:  {np.max(all_agent_uncertainties):.4f}")

        if all_user_uncertainties:
            console.print("\n[bold]User Confusion (U_i,user):[/bold]")
            console.print(f"  Mean: {np.mean(all_user_uncertainties):.4f}")
            console.print(f"  Std:  {np.std(all_user_uncertainties):.4f}")
            console.print(f"  Min:  {np.min(all_user_uncertainties):.4f}")
            console.print(f"  Max:  {np.max(all_user_uncertainties):.4f}")
    
    # Semantic distance metrics
    all_da_scores = []
    all_do_scores = []
    all_do_agent = []
    all_do_user = []
    
    for sim in analysis.results:
        for score in sim.uncertainty_scores:
            if score.da_score is not None:
                all_da_scores.append(score.da_score)
            if score.do_score is not None:
                all_do_scores.append(score.do_score)
                if score.do_type == "agent_coherence":
                    all_do_agent.append(score.do_score)
                elif score.do_type == "user_coherence":
                    all_do_user.append(score.do_score)
    
    if all_da_scores or all_do_scores:
        console.print("\n[bold cyan]Semantic Distance Metrics[/bold cyan]")
        
        if all_da_scores:
            console.print("\n[bold]Inquiry Drift (Da):[/bold]")
            console.print(f"  Mean: {np.mean(all_da_scores):.4f}")
            console.print(f"  Std:  {np.std(all_da_scores):.4f}")
            console.print(f"  Min:  {np.min(all_da_scores):.4f}")
            console.print(f"  Max:  {np.max(all_da_scores):.4f}")
            console.print(f"  Count: {len(all_da_scores)}")
        
        if all_do_scores:
            console.print("\n[bold]Inference Gap (Do):[/bold]")
            console.print(f"  Mean: {np.mean(all_do_scores):.4f}")
            console.print(f"  Std:  {np.std(all_do_scores):.4f}")
            console.print(f"  Min:  {np.min(all_do_scores):.4f}")
            console.print(f"  Max:  {np.max(all_do_scores):.4f}")
            console.print(f"  Count: {len(all_do_scores)}")
            
            if all_do_agent:
                console.print(f"\n  [dim]Agent Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_agent):.4f}")
                console.print(f"    Count: {len(all_do_agent)}")
            
            if all_do_user:
                console.print(f"\n  [dim]User Coherence:[/dim]")
                console.print(f"    Mean: {np.mean(all_do_user):.4f}")
                console.print(f"    Count: {len(all_do_user)}")
    
    # SAUP-D Aggregation Scores
    all_saup_scores = []
    saup_passed = []
    saup_failed = []
    
    for sim in analysis.results:
        if sim.saup_metrics is not None:
            saup_score = sim.saup_metrics.get('saup_score')
            if saup_score is not None:
                all_saup_scores.append(saup_score)
                
                # Track by ground truth
                if sim.ground_truth_pass is not None:
                    if sim.ground_truth_pass:
                        saup_passed.append(saup_score)
                    else:
                        saup_failed.append(saup_score)
    
    if all_saup_scores:
        console.print("\n[bold cyan]SAUP-D Aggregation Scores[/bold cyan]")
        console.print(f"  Mean SAUP: {np.mean(all_saup_scores):.4f}")
        console.print(f"  Std SAUP:  {np.std(all_saup_scores):.4f}")
        console.print(f"  Min SAUP:  {np.min(all_saup_scores):.4f}")
        console.print(f"  Max SAUP:  {np.max(all_saup_scores):.4f}")
        console.print(f"  Total simulations: {len(all_saup_scores)}")
        
        # Ground truth correlation
        if saup_passed or saup_failed:
            total_with_gt = len(saup_passed) + len(saup_failed)
            console.print(f"\n  Ground Truth: {len(saup_passed)}/{total_with_gt} passed ({100*len(saup_passed)/total_with_gt:.1f}%)")
            if saup_passed:
                console.print(f"  Mean SAUP (passed): {np.mean(saup_passed):.4f}")
            if saup_failed:
                console.print(f"  Mean SAUP (failed): {np.mean(saup_failed):.4f}")
    
    # AUROC Evaluation (Failure Prediction)
    if analysis.auroc_metrics is not None:
        auroc = analysis.auroc_metrics
        console.print("\n[bold cyan]AUROC Evaluation (Failure Prediction)[/bold cyan]")
        console.print(f"  Hypothesis: High SAUP-D → High probability of failure")
        console.print(f"  Dataset: {auroc.num_samples} tasks ({auroc.num_failures} failures, {auroc.num_successes} successes)")
        console.print()
        
        # AUROC interpretation
        auroc_value = auroc.auroc
        if auroc_value > 0.9:
            auroc_color = "green"
            auroc_label = "Excellent"
        elif auroc_value > 0.8:
            auroc_color = "green"
            auroc_label = "Good"
        elif auroc_value > 0.7:
            auroc_color = "yellow"
            auroc_label = "Fair"
        elif auroc_value > 0.6:
            auroc_color = "yellow"
            auroc_label = "Poor"
        else:
            auroc_color = "red"
            auroc_label = "Very Poor"
        
        console.print(f"  AUROC: [{auroc_color}]{auroc_value:.4f}[/{auroc_color}] ({auroc_label})")
        console.print(f"  Optimal Threshold: {auroc.optimal_threshold:.4f}")
        console.print(f"  Accuracy: {auroc.accuracy:.4f}")
        console.print(f"  Precision: {auroc.precision:.4f}")
        console.print(f"  Recall: {auroc.recall:.4f}")
        console.print(f"  F1 Score: {auroc.f1_score:.4f}")
        
        # Interpretation
        console.print()
        if auroc_value > 0.7:
            console.print(f"  [green]✅ SAUP-D has {auroc_label.lower()} predictive power for task failure![/green]")
            console.print(f"  [green]   Tasks with SAUP-D > {auroc.optimal_threshold:.4f} are at high risk of failure.[/green]")
        elif auroc_value > 0.6:
            console.print(f"  [yellow]⚠️  SAUP-D shows fair predictive power (AUROC={auroc_value:.4f}).[/yellow]")
            console.print(f"  [yellow]   Consider combining with other features or tuning α,β,γ weights.[/yellow]")
        else:
            console.print(f"  [red]❌ SAUP-D shows poor predictive power (AUROC={auroc_value:.4f}).[/red]")
            if auroc.num_samples < 30:
                console.print(f"  [yellow]   Note: Small sample size ({auroc.num_samples}) limits statistical power.[/yellow]")
            else:
                console.print(f"  [yellow]   The hypothesis may not hold for this dataset/configuration.[/yellow]")

    # Per-simulation summary
    console.print("\n[bold cyan]Per-Simulation Summary[/bold cyan]\n")
    for i, sim in enumerate(analysis.results[:5]):  # Show first 5
        summary = sim.summary
        console.print(f"[bold]Simulation {i+1}[/bold] (Task: {sim.task_id})")
        console.print(f"  Turns: {sim.turn_count}")
        console.print(
            f"  Mean uncertainty (overall): {summary.get('mean_uncertainty_overall', 0):.4f}"
        )
        console.print(
            f"  Mean uncertainty (agent):   {summary.get('mean_uncertainty_agent', 0):.4f}"
        )
        console.print(
            f"  Mean uncertainty (user):    {summary.get('mean_uncertainty_user', 0):.4f}"
        )
        
        # Display SAUP score if available
        if sim.saup_metrics:
            console.print(f"  SAUP Score: {sim.saup_metrics['saup_score']:.4f}")
            console.print(f"  Ground Truth: {'✅ Pass' if sim.ground_truth_pass else '❌ Fail' if sim.ground_truth_pass is not None else 'N/A'}\n")
        else:
            console.print()

    if len(analysis.results) > 5:
        console.print(f"... and {len(analysis.results) - 5} more simulations\n")

    console.print("=" * 80, style="dim")


def print_detailed_trajectory(
    analysis: UncertaintyAnalysis, sim_index: int, console: Console
):
    """Print detailed turn-by-turn uncertainty for a specific simulation."""
    if sim_index >= len(analysis.results):
        console.print(
            f"[yellow]⚠️  Simulation index {sim_index} not found.[/yellow]"
        )
        return

    sim = analysis.results[sim_index]

    console.print("\n" + "=" * 80, style="dim")
    console.print(
        f"DETAILED TRAJECTORY: Simulation {sim_index + 1}",
        style="bold cyan",
        justify="center",
    )
    console.print("=" * 80, style="dim")
    console.print(f"\n[bold]Simulation ID:[/bold] {sim.simulation_id}")
    console.print(f"[bold]Task ID:[/bold] {sim.task_id}")
    console.print(f"[bold]Trial:[/bold] {sim.trial}")
    console.print(f"[bold]Total Turns:[/bold] {sim.turn_count}")
    
    # Display SAUP score if available
    if sim.saup_metrics:
        console.print(f"[bold]SAUP Score:[/bold] {sim.saup_metrics['saup_score']:.4f}")
        console.print(f"[bold]Ground Truth:[/bold] {'✅ Pass' if sim.ground_truth_pass else '❌ Fail' if sim.ground_truth_pass is not None else 'N/A'}")
    console.print()

    # Calculate weights for display
    weights = []
    if sim.saup_metrics:
        config = SAUPConfig()
        for score in sim.uncertainty_scores:
            from tau2.metrics.uncertainty import calculate_situational_weight
            w = calculate_situational_weight(
                score.da_score,
                score.do_score if score.do_type == "agent_coherence" else None,
                score.do_score if score.do_type == "user_coherence" else None,
                config
            )
            weights.append(w)

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Turn", style="dim", width=5)
    table.add_column("Actor", width=8)
    table.add_column("U_i", justify="right", width=8)
    table.add_column("Da", justify="right", width=8)
    table.add_column("Do", justify="right", width=8)
    table.add_column("W_i", justify="right", width=8)
    table.add_column("Do Type", width=10)
    table.add_column("Content", width=35)

    for idx, score in enumerate(sim.uncertainty_scores):
        # Color code by uncertainty level
        if score.ui_score < 0.1:
            ui_color = "green"
        elif score.ui_score < 0.3:
            ui_color = "yellow"
        else:
            ui_color = "red"
        
        # Color code Da (inquiry drift)
        da_str = "—"
        if score.da_score is not None:
            if score.da_score < 0.3:
                da_color = "green"
            elif score.da_score < 0.6:
                da_color = "yellow"
            else:
                da_color = "red"
            da_str = f"[{da_color}]{score.da_score:.3f}[/{da_color}]"
        
        # Color code Do (inference gap)
        do_str = "—"
        if score.do_score is not None:
            if score.do_score < 0.3:
                do_color = "green"
            elif score.do_score < 0.6:
                do_color = "yellow"
            else:
                do_color = "red"
            do_str = f"[{do_color}]{score.do_score:.3f}[/{do_color}]"
        
        # Display weight if available
        w_str = "—"
        if weights and idx < len(weights):
            w = weights[idx]
            if w < 0.3:
                w_color = "green"
            elif w < 0.6:
                w_color = "yellow"
            else:
                w_color = "red"
            w_str = f"[{w_color}]{w:.3f}[/{w_color}]"
        
        do_type_str = score.do_type[:10] if score.do_type else "—"

        table.add_row(
            str(score.turn),
            score.actor,
            f"[{ui_color}]{score.ui_score:.3f}[/{ui_color}]",
            da_str,
            do_str,
            w_str,
            do_type_str,
            score.content_preview[:35] + "...",
        )

    console.print(table)
    console.print("\n" + "=" * 80, style="dim")


def main():
    """Main entry point for the uncertainty analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze Tau-2 trajectories and calculate uncertainty metrics"
    )
    parser.add_argument(
        "simulation_file", type=str, help="Path to the simulation results JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path (optional, defaults to data/uncertainty/<same_filename>)",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include detailed statistics in output",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Print detailed turn-by-turn trajectory",
    )
    parser.add_argument(
        "--sim-index",
        type=int,
        default=0,
        help="Simulation index to show in detailed view (default: 0)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file",
    )
    parser.add_argument(
        "--saup-config",
        type=json.loads,
        default='{"alpha": 1.0, "beta": 1.0, "gamma": 1.0}',
        help='SAUP-D weight configuration (JSON format, e.g., \'{"alpha": 1.0, "beta": 1.0, "gamma": 1.0}\')',
    )
    parser.add_argument(
        "--no-auroc",
        action="store_true",
        help="Skip AUROC calculation (useful if ground truth not available)",
    )

    args = parser.parse_args()

    console = Console()

    # Check if file exists
    sim_path = Path(args.simulation_file)
    if not sim_path.exists():
        console.print(
            f"[red]❌ Error: File not found: {args.simulation_file}[/red]"
        )
        sys.exit(1)

    try:
        # Load simulation data
        console.print(
            f"[cyan]📂 Loading simulation results from: {args.simulation_file}[/cyan]"
        )
        results = Results.load(sim_path)

        # Analyze
        console.print("[cyan]🔄 Analyzing trajectories and calculating uncertainty scores...[/cyan]")
        analysis = analyze_results(
            results,
            verbose=args.verbose,
            calculate_auroc=not args.no_auroc
        )

        # Print summary
        print_summary(analysis, console)

        # Print detailed trajectory if requested
        if args.detailed:
            print_detailed_trajectory(analysis, args.sim_index, console)

        # Determine output path
        if not args.no_save:
            if args.output:
                # User specified custom output path
                output_path = Path(args.output)
            else:
                # Default: save to data/uncertainty with same filename
                sim_filename = sim_path.name
                # Get the project root (assuming script is in src/tau2/scripts/)
                project_root = Path(__file__).parent.parent.parent.parent
                output_dir = project_root / "data" / "uncertainty"
                output_path = output_dir / sim_filename
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the analysis
            with open(output_path, "w") as f:
                json.dump(analysis.model_dump(), f, indent=2)

            console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")

        console.print("\n[green]✅ Analysis complete![/green]")

    except json.JSONDecodeError as e:
        console.print(f"[red]❌ Error: Invalid JSON file: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error processing file: {e}[/red]")
        logger.exception("Error during uncertainty analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()

