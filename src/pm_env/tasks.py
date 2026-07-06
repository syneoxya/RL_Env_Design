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
                        Build a CPU-compatible FlowHFT-style policy for high-frequency market
                        making.

                        Your task is to build a compact policy that uses the same main idea in a
                        lightweight benchmark:
                        learn a conditional flow over quote actions from expert demonstrations, then
                        use the learned flow to produce bid/ask offsets from the current market
                        state.

                        The intended policy is:
                        1. Start from a simple normalized action, such as zeros or fixed Gaussian
                           prior mean.
                        2. Condition on the current 10-dimensional market state.
                        3. Integrate a learned vector field for a few Euler steps.
                        4. Denormalize the generated action into bid/ask quote offsets.
                        5. Apply a small affine calibration to improve rollout behavior.

                        This benchmark uses a compact single-step action interface. The policy
                        receives a 10-dimensional market state and outputs two positive quote
                        offsets:

                            state -> [bid_offset, ask_offset]

                        Data location:

                            {env_dir}

                        Visible files:
                        - train_states.npy: training states, shape (N, 10)
                        - train_actions.npy: expert bid/ask offsets, shape (N, 2)
                        - public_val_states.npy: public validation states, shape (M, 10)
                        - public_val_actions.npy: public validation expert offsets, shape (M, 2)
                        - public_val_paths.npz: public market paths for rollout calibration
                        - public_val_regimes.npy: public validation regime IDs
                        - normalization_stats.npz: state/action summary statistics
                        - data_card.json and README_DATA.md: task metadata

                        The visible demonstrations come from analytical expert strategies used as
                        FlowHFT instructors:
                        - Avellaneda-Stoikov style expert
                        - GLFT style expert
                        - GLFT with drift style expert

                        PPO is intentionally removed from this benchmark so the solution remains
                        CPU-compatible. Hidden scoring uses unseen market paths and unseen regimes.
                        You cannot see hidden paths or hidden regime parameters.

                        State and action definitions:
                        The input state is a 10-dimensional market observation:
                        0. inventory / max_inventory
                        1. time fraction
                        2. one-step return
                        3. recent return mean
                        4. recent realized volatility
                        5. recent buy arrivals / 50
                        6. recent sell arrivals / 50
                        7. order-flow imbalance
                        8. price deviation from start
                        9. remaining time

                        The output action has two positive quote offsets:
                        - bid_offset: distance below mid-price for the bid quote
                        - ask_offset: distance above mid-price for the ask quote

                        Offsets are fractions of the current mid-price. For example, 0.02 means
                        quote about 2 percent away from mid-price. Practical offsets are positive
                        and usually in the rough range 0.002 to 0.250.

                        Required final files:

                        1. {module_path}

                           This file must define:

                               class FlowHFTPolicy(torch.nn.Module)

                           Requirements:
                           - FlowHFTPolicy must be a torch.nn.Module.
                           - FlowHFTPolicy.__init__ must not require any arguments.
                           - FlowHFTPolicy.forward(x) must accept a float tensor with shape
                             (batch, 10).
                           - FlowHFTPolicy.forward(x) must return a float tensor with shape
                             (batch, 2).
                           - The two outputs are [bid_offset, ask_offset].
                           - Outputs must be finite and positive.
                           - The model must run on CPU.
                           - Inference must not load data, require command-line arguments, use CUDA,
                             access the internet, or access hidden files.

                        2. {checkpoint_path}

                           This file must be a state_dict that loads into FlowHFTPolicy:

                               model = FlowHFTPolicy()
                               model.load_state_dict(torch.load(..., map_location="cpu"))

                        Flow policy to implement:

                        Train a conditional vector field:

                            v_theta(a_t, t | O_t)

                        where:
                        - O_t is the current market observation
                        - a_0 is an initial action sample from a simple prior, such as Gaussian
                          noise in normalized action space
                        - a_E is the expert action from the demonstrations
                        - t is sampled uniformly from [0, 1]
                        - a_t = (1 - t) * a_0 + t * a_E
                        - the target vector field is u_t = a_E - a_0

                        Train the vector field with the flow matching loss:

                            L_FM = mean_squared_error(v_theta(a_t, t | O_t), a_E - a_0)

                        This teaches the model to transport noisy actions toward expert-like
                        bid/ask offsets conditioned on market state.

                        Recommended final policy architecture:

                        Implement FlowHFTPolicy as a compact conditional flow model. A strong
                        CPU-friendly design is:
                        - register buffers for state_mean, state_std, action_mean, and action_std
                        - normalize input states inside forward
                        - use a small state encoder MLP for the 10 state features
                        - use a small action-time encoder for the current normalized action a_t
                          and scalar time t
                        - combine state, action, and time features in a vector-field MLP
                        - output a 2-dimensional velocity vector for bid/ask action space
                        - integrate the vector field with a small fixed number of Euler steps
                        - denormalize the generated action back to quote-offset units
                        - apply the fine-tuned affine calibration
                        - apply a positive/safe transform or clamp so offsets are finite and
                          positive

                        The final FlowHFTPolicy should actually use the learned vector field at
                        inference time. Do not replace the flow with a plain state -> action
                        supervised MLP. A direct action predictor may be used only as an auxiliary
                        training signal, but the submitted forward pass should generate actions by
                        integrating v_theta(a_t, t | O_t).

                        The forward method should be deterministic. Do not sample random noise in
                        forward. Use a fixed initial action such as zeros in normalized action
                        space:

                            z = zeros(batch, 2)
                            for k in range(num_euler_steps):
                                t = k / num_euler_steps
                                z = z + (1 / num_euler_steps) * v_theta(z, t | state)
                            raw_action = z * action_std + action_mean
                            final_action = alpha * raw_action + beta
                            final_action = positive_and_safe_transform(final_action)

                        Use a small number of Euler steps such as 4, 6, 8, or 10. Hidden rollout
                        calls the policy many times, so inference must be fast.

                        Output scaling guidance:
                        - Expert quote offsets are usually near a few percent, often around
                          0.02 to 0.08 depending on regime.
                        - Do not map denormalized offsets through a wide transform such as
                          0.5 * sigmoid(offset) + 0.001; that tends to produce quotes near 0.25
                          before training and can make the policy too wide or poorly calibrated.
                        - Prefer generating actions in normalized action space, denormalizing with
                          action_std and action_mean, applying alpha/beta calibration, then clamping
                          to a reasonable positive range such as [0.002, 0.250].
                        - If you use softplus, apply it in a way that preserves the expert offset
                          scale instead of pushing typical outputs far above the data mean.

                        Fine-tuning component:

                        Add a lightweight affine action calibration:

                            calibrated_action = alpha * raw_action + beta

                        where alpha is a scalar or 2-dimensional scale and beta is a 2-dimensional
                        offset for [bid_offset, ask_offset]. Calibrate alpha and beta using only
                        visible data. Good CPU-friendly options:
                        - grid search alpha values around 1.0
                        - grid search small bid/ask beta offsets
                        - use public_val_actions.npy for imitation sanity checks
                        - use public_val_paths.npz for rollout-oriented calibration if practical
                        - choose parameters that improve PnL/risk behavior without collapsing
                          action diversity

                        Store the selected alpha and beta as parameters or buffers in the final
                        FlowHFTPolicy checkpoint.

                        Calibration objective:
                        Do not choose alpha and beta by imitation MSE or final PnL alone. The
                        hidden scorer rewards a balanced market-making policy that is competitive
                        with AS, GLFT, and GLFT-drift experts across market regimes. During
                        calibration, prefer settings that improve a score-shaped objective on
                        visible validation rollouts:
                        - high mean normalized PnL versus the expert family
                        - stable step returns and high Sharpe ratio versus the expert family
                        - low maximum drawdown relative to the expert family
                        - low average and maximum absolute inventory relative to the expert family
                        - positive PnL across most validation regimes, not only one regime
                        - enough action variation to be adaptive, without noisy extreme quotes

                        If you use public_val_paths.npz for calibration, inspect its array names and
                        build a small rollout check from the visible paths. Evaluate candidate
                        calibration grids by regime using public_val_regimes.npy. Penalize candidates
                        with high regime-to-regime PnL variation, negative validation regimes, or
                        large inventory accumulation even if their average PnL is high.

                        Store the selected alpha and beta as parameters or buffers in the final
                        FlowHFTPolicy checkpoint.

                        Training guidance:
                        - Train in normalized action space using action statistics from
                          normalization_stats.npz or from the visible training data.
                        - Train the conditional flow field on train_states.npy and
                          train_actions.npy.
                        - Do not stop at ordinary supervised imitation. If you add a direct
                          supervised action head, use it only as an auxiliary/distillation loss;
                          the submitted forward pass should still integrate the flow field.
                        - Save the checkpoint only after verifying that FlowHFTPolicy.forward()
                          runs the same architecture that was trained with the flow-matching loss.
                        - Keep the model compact enough for CPU inference.
                        - Do not use BatchNorm; hidden rollout may call the model with batch size 1.
                        - Do not use Dropout; final evaluation should be deterministic.
                        - LayerNorm, residual MLP blocks, SiLU/ReLU activations, time embeddings,
                          and compact vector-field networks are acceptable.

                        Market-making behavior the policy should learn:
                        - quote wider when realized volatility or adverse selection risk is high
                        - adapt to buy/sell arrival rates and order-flow imbalance
                        - skew quotes to control inventory
                        - when inventory is positive, reduce further buying and encourage selling
                        - when inventory is negative, reduce further selling and encourage buying
                        - remain adaptive across high/low volatility and high/low demand regimes
                        - avoid nearly constant quote offsets

                        Hidden judge objective:
                        Hidden scoring uses rollout behavior, not imitation loss alone. The final
                        score is weighted approximately as:
                        - 0.30 regime-level normalized PnL versus AS/GLFT/GLFT-drift experts
                        - 0.20 regime-level Sharpe ratio versus AS/GLFT/GLFT-drift experts
                        - 0.15 regime-level maximum drawdown control versus those experts
                        - 0.15 regime-level average and maximum inventory control versus those experts
                        - 0.10 robustness across hidden regimes
                        - 0.10 action adaptivity

                        This means an aggressive policy can score poorly even with good average PnL
                        if it has unstable returns, high inventory, one bad regime, or insufficient
                        action adaptivity. Prefer robust market-making behavior over a policy that
                        earns most of its profit by carrying large directional inventory.

                        The hidden scorer compares against AS, GLFT, and GLFT-drift as fixed
                        regime-level baselines. It does not use a hindsight oracle that picks the
                        best expert after each individual episode. The goal is to learn a single
                        adaptive FlowHFT policy that is competitive with the expert family across
                        regimes.

                        Validation targets before final submission:
                        - FlowHFTPolicy loads from the checkpoint.
                        - Outputs are finite and positive.
                        - Outputs are not nearly constant over public_val_states.npy.
                        - Predictions have reasonable imitation error on public_val_actions.npy.
                        - Bid/ask skew responds sensibly to inventory.
                        - Quote widths respond sensibly to volatility and order-flow conditions.
                        - Action standard deviation across public validation states is meaningful;
                          a useful target range is roughly 0.015 to 0.030 average std across bid
                          and ask offsets. Do not create variation with random noise.
                        - Public validation rollout, if implemented, has positive PnL in most
                          regimes and does not depend on a single regime for most of the profit.
                        - Average absolute inventory remains controlled. If inventory grows while
                          PnL improves, prefer a safer calibration unless the Sharpe and drawdown
                          also clearly improve.
                        - Sharpe-oriented stability matters: avoid quote settings that produce rare
                          large wins but frequent noisy losses.

                        Integrity rules / no reward hacking:
                        - Solve the ML task directly. Do not try to exploit the environment,
                          scorer, file system, hidden data layout, or validation artifacts.
                        - Do not read, modify, delete, rename, copy, or infer hidden scoring data.
                        - Do not modify the scorer, task files, setup files, package files, run
                          config, generated data files, or any files outside the required final
                          artifacts.
                        - Do not monkeypatch Python, torch, numpy, import machinery, random seeds,
                          file APIs, environment variables, or subprocess behavior to influence the
                          judge.
                        - Do not save final files anywhere except the required paths.
                        - Do not hardcode predictions, labels, row indices, file hashes, public
                          validation examples, episode IDs, regime IDs, filenames, path strings, or
                          file-specific behavior.
                        - Do not special-case public validation files or memorize validation rows.
                          Public validation data may be used only for honest training diagnostics
                          and calibration.
                        - Do not make the policy depend on the current working directory, absolute
                          paths, clock time, process ID, machine details, hidden file existence, or
                          any other side channel.
                        - Do not use internet downloads.
                        - Use only installed packages.
                        - Do not run background jobs with &.
                        - Do not use large transformers or heavy external frameworks.
                        - Do not create extra final artifacts, markdown reports, or long
                          summaries. A short final stdout summary of validation metrics is allowed.

                        Required implementation order:
                        1. Read the visible metadata and arrays.
                        2. Write {module_path} with a valid FlowHFTPolicy class.
                        3. Train the conditional flow matching vector field on visible data only.
                        4. Calibrate alpha and beta using only visible validation data.
                        5. Save a state_dict to {checkpoint_path}.
                        6. Reload the checkpoint into FlowHFTPolicy.
                        7. Print a short final stdout summary with validation imitation MSE,
                           action std, output min/max, and any visible rollout score you computed.
                        8. Run the final verification script below.
                        9. Stop immediately after verification passes.

                        Artifact safety:
                        - If you edit policy.py after saving the checkpoint, immediately save a
                          matching checkpoint.
                        - Do not leave train-only classes or architectures that differ from the
                          final FlowHFTPolicy class.
                        - The final artifact must be importable from policy.py without requiring
                          data loading, CUDA, internet access, or command-line arguments.

                        Final verification script:

                            python3 - << 'EOF'
                            import torch
                            import importlib.util

                            module_path = "{module_path}"
                            checkpoint_path = "{checkpoint_path}"

                            spec = importlib.util.spec_from_file_location("policy", module_path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            model = module.FlowHFTPolicy()

                            params = sum(p.numel() for p in model.parameters())
                            print("parameter_count", params)

                            state = torch.load(checkpoint_path, map_location="cpu")
                            model.load_state_dict(state)

                            model.eval()
                            x = torch.randn(4, 10)
                            with torch.no_grad():
                                y = model(x)

                            assert y.shape == (4, 2), y.shape
                            assert torch.isfinite(y).all(), "non-finite outputs"
                            assert (y > 0).all(), "quote offsets must be positive"

                            print("output_min", y.min().item())
                            print("output_max", y.max().item())
                            print("verification passed")
                            EOF

                        Final stop rule:
                        After the verification script prints "verification passed", stop
                        immediately. Do not print policy.py, create long summaries, retrain,
                        rerun diagnostics, or modify policy.py or flowhft_policy.pt.
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
