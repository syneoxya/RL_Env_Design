# src/pm_env/tasks.py

import sys
from pathlib import Path
from textwrap import dedent

from pm_env.get_data_dir import get_env_data_dir
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Step, Task


def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    module_name = "policy.py"
    checkpoint_name = "flowhft_policy.pt"
    score_script = (Path(__file__).parent / "scoring_script.py").as_posix()

    env_dir = get_env_data_dir()
    module_path = f"{env_dir}/{module_name}"
    checkpoint_path = f"{env_dir}/{checkpoint_name}"

    return [
        Task(
            id="flowhft-cpu-market-making",
            tools=["bash"],
            steps=[
                Step(
                    instructions=dedent(
                        f"""
                        # CPU FlowHFT market-making policy

                        Build a compact conditional flow policy that maps a 10D market state to
                        positive bid/ask quote offsets. Train and calibrate it using only the
                        visible files in:

                            {env_dir}

                        ## Data

                        - train_states.npy: (N, 10)
                        - train_actions.npy: expert offsets, (N, 2)
                        - public_val_states.npy and public_val_actions.npy
                        - public_val_paths.npz and public_val_regimes.npy for calibration
                        - normalization_stats.npz, data_card.json, and README_DATA.md

                        State features, in order, are normalized inventory, time fraction,
                        one-step return, recent return mean, realized volatility, recent buy and
                        sell arrivals, order-flow imbalance, price deviation, and remaining time.
                        Actions are [bid_offset, ask_offset], expressed as positive fractions of
                        mid-price; the safe range is [0.002, 0.250].

                        ## Required flow training

                        In normalized action space, sample a base action a_0 and t ~ Uniform(0, 1):

                            a_t = (1 - t) * a_0 + t * a_E
                            target = a_E - a_0
                            L_FM = MSE(v_theta(a_t, t | state), target)

                        Train this conditional vector field on train_states.npy and
                        train_actions.npy. A supervised head may be auxiliary, but the submitted
                        forward pass must generate actions by integrating the learned flow.

                        ## Required artifacts and executable contract

                        Create only:

                        1. {module_path}
                           - defines FlowHFTPolicy(torch.nn.Module)
                           - __init__ takes no arguments
                           - forward accepts float (batch, 10) and returns finite, positive
                             (batch, 2) CPU tensors
                           - inference is deterministic and self-contained

                        2. {checkpoint_path}
                           - a state_dict that loads strictly into FlowHFTPolicy on CPU

                        FlowHFTPolicy must:

                        - register buffers state_mean/state_std with shape (10,)
                        - register buffers action_mean/action_std, alpha, and beta with shape (2,)
                        - keep both standard-deviation buffers positive
                        - expose vector_field as a torch.nn.Module
                        - implement velocity(action_t, time, normalized_state) -> (batch, 2)
                        - implement raw_action(state) -> (batch, 2) by integrating vector_field
                          from a fixed initial action for a small number of Euler steps
                        - compute forward as clamp(alpha * raw_action(state) + beta, 0.002, 0.250),
                          with safe handling of non-finite values

                        Do not replace the flow with a direct state-to-action MLP. Avoid BatchNorm,
                        Dropout, CUDA, random sampling during inference, or inference-time file I/O.

                        ## Affine calibration

                        Select alpha and beta using only visible validation data. Prefer a small
                        grid evaluated on public_val_paths.npz across regimes. Balance normalized
                        PnL, Sharpe, drawdown, average/maximum absolute inventory, positive-regime
                        coverage, and action variation; do not optimize imitation MSE or final PnL
                        alone. Save alpha and beta in the checkpoint.

                        ## Hidden score

                        Hidden paths and regimes compare the policy with AS, GLFT, and GLFT-drift:

                        - 30% regime-level normalized PnL
                        - 20% Sharpe ratio
                        - 15% maximum drawdown control
                        - 15% average/maximum inventory control
                        - 10% robustness across regimes
                        - 10% deterministic action adaptivity

                        A strong policy widens with risk, uses bid/ask skew to control inventory,
                        adapts to volatility and order flow, and avoids nearly constant actions.

                        ## Integrity

                        Solve the ML task directly. Do not access or infer hidden data; modify the
                        scorer, task, setup, package, run configuration, or generated datasets;
                        monkeypatch runtime behavior; hardcode examples or file-specific behavior;
                        use side channels or internet downloads; or write final artifacts outside
                        the two required paths. Public validation data may be used only for honest
                        diagnostics and calibration.

                        ## Finish

                        Train, calibrate, save, reload, and verify the exact final architecture.
                        If policy.py changes, regenerate its matching checkpoint. Run:

                            python3 - << 'EOF'
                            import importlib.util
                            import torch

                            module_path = "{module_path}"
                            checkpoint_path = "{checkpoint_path}"
                            spec = importlib.util.spec_from_file_location("policy", module_path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            model = module.FlowHFTPolicy()
                            state = torch.load(checkpoint_path, map_location="cpu")
                            model.load_state_dict(state, strict=True)
                            model.eval()
                            with torch.no_grad():
                                y = model(torch.randn(4, 10))

                            assert y.shape == (4, 2)
                            assert torch.isfinite(y).all() and (y > 0).all()
                            required = {{"state_mean", "state_std", "action_mean",
                                        "action_std", "alpha", "beta"}}
                            assert required <= set(dict(model.named_buffers()))
                            print("verification passed")
                            EOF

                        After verification passes, stop. Print only a short summary of validation
                        MSE, action standard deviation, output range, and visible rollout score.
                        """
                    ),
                    judge=ExecutableJudge(
                        [
                            sys.executable,
                            score_script,
                            module_path,
                            checkpoint_path,
                            "/tmp/flowhft_results.txt",
                        ]
                    ),
                )
            ],
        )
    ]
