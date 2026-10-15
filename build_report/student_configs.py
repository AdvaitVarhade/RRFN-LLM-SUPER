"""
Student Configuration Profiles and Project Work Division
Defines the technical pillars, module ownership, and personal details for all 3 team members
in compliance with VIT Chennai BCSE497J Project Report Guidelines.
"""

STUDENT_PROFILES = [
    {
        "id": "advait",
        "name": "ADVAIT RAVINDRAKUMAR VARHADE",
        "reg_no": "23BCE5098",
        "file_prefix": "project_report_Advait_Varhade_23BCE5098",
        "pillar_title": "Mathematical Noise-Transition Modeling, Tikhonov Operator Inversion, Risk-Consistent Optimization & Theoretical Proofs",
        "short_pillar": "Theoretical Risk Consistency & Noise Inversion",
        "role_summary": "Lead Theoretical Foundations & Mathematical Optimization Architect",
        "modules_owned": [
            "rrfn/transition_matrix.py",
            "rrfn/anchor_selector.py",
            "rrfn/risk_loss.py",
            "models/softmax_classifier.py"
        ],
        "key_responsibilities": (
            "Formulation and mathematical proof of Theorem 3.1 (Tikhonov spectral operator norm bound $\\|T_\\lambda^\\dagger\\|_2 \\le \\frac{1}{2\\sqrt{\\lambda}}$), "
            "Theorem 3.2 (surrogate excess clean risk consistency $\\mathcal{E}^*(\\hat{f}) \\le \\frac{L_{\\mathrm{CE}}}{\\sigma_0} \\delta + \\mathcal{O}(1/\\sqrt{N})$), consensus anchor selection, "
            "Riesz-Thorin Lipschitz gradient bounds, and overall system theoretical stability."
        ),
        "teammates": [
            {
                "name": "P. MANAS",
                "reg_no": "23BCE5076",
                "pillar": "Multi-View Interaction Reliability Filter (MIRF), LLM Review Auditing & 2D BLAS Review-Bombing Detection"
            },
            {
                "name": "VIVAASH KOUNDAL",
                "reg_no": "23BCE5095",
                "pillar": "Risk-Aware Graph Contrastive Learning, Denoised SUPER Recommender Engine & Calibrated Pareto Debiasing"
            }
        ],
        "abstract_focus": (
            "With primary focus on theoretical foundations and mathematical guarantees, this report establishes that Tikhonov-regularized "
            "noise inversion strictly upper-bounds operator norms by $\\|T_\\lambda^\\dagger\\|_2 \\le \\frac{1}{2\\sqrt{\\lambda}}$, guaranteeing non-singular gradient "
            "stability without numerical divergence, and proves asymptotic surrogate risk consistency $\\mathcal{O}(\\delta + 1/\\sqrt{N})$ alongside confidence "
            "inversion resolvent bounds $\\mathcal{O}(\\delta/\\sqrt{\\lambda} + \\sqrt{\\lambda})$."
        ),
        "future_work_focus": (
            "Non-linear stochastic rating transition kernels using neural optimal transport, and formal differential privacy (DP-SGD) "
            "guarantees protecting user rating profiles against reconstruction attacks."
        )
    },
    {
        "id": "manas",
        "name": "P. MANAS",
        "reg_no": "23BCE5076",
        "file_prefix": "project_report_Manas_23BCE5076",
        "pillar_title": "Multi-View Interaction Reliability Filter (MIRF), Asynchronous LLM Review Auditing & Hardware-Accelerated 2D BLAS Burst Detection",
        "short_pillar": "MIRF Multi-Modal Auditing & 2D BLAS Burst Detection",
        "role_summary": "Lead Multi-Modal Reliability Auditing & Security Defense Architect",
        "modules_owned": [
            "fusion/reliability_fusion.py",
            "llm_auditor/auditor.py",
            "llm_auditor/cache.py",
            "llm_auditor/prompt_builder.py",
            "bombing_detector/bomb_scorer.py",
            "bombing_detector/temporal_density.py",
            "bombing_detector/polarity_skew.py"
        ],
        "key_responsibilities": (
            "Architecture and implementation of the Multi-view Interaction Reliability Filter (MIRF) tri-view synthesis ($R_{\\mathrm{RRFN}}, R_{\\mathrm{LLM}}, R_{\\mathrm{bomb}}$), "
            "asynchronous LLM semantic review auditing using persistent SQLite Write-Ahead Logging (WAL) caching for $\\mathcal{O}(1)$ retrieval, "
            "hardware-accelerated 2D BLAS Level-3 GEMM review-bombing burst detection (<5ms latency for 42,000 interactions; $>500\\times$ speedup), "
            "and attack detection vs. authentic retention evaluation (Attack AP vs Authentic AP)."
        ),
        "teammates": [
            {
                "name": "ADVAIT RAVINDRAKUMAR VARHADE",
                "reg_no": "23BCE5098",
                "pillar": "Mathematical Noise-Transition Modeling, Tikhonov Operator Inversion & Risk-Consistent Proofs"
            },
            {
                "name": "VIVAASH KOUNDAL",
                "reg_no": "23BCE5095",
                "pillar": "Risk-Aware Graph Contrastive Learning, Denoised SUPER Recommender Engine & Calibrated Pareto Debiasing"
            }
        ],
        "abstract_focus": (
            "With primary focus on multi-modal interaction auditing and real-time anomaly filtering, this report details the engineering of the "
            "Multi-view Interaction Reliability Filter (MIRF). By synthesizing statistical model prediction consistency, asynchronous Large Language "
            "Model (LLM) review auditing with thread-safe SQLite WAL caching, and 2D BLAS vectorized burst scoring (achieving 4.95 ms latency for 42,000 "
            "interactions, a $>500\\times$ speedup), MIRF achieves Attack Detection AP of 0.6884 on Amazon and 0.6816 on Yelp (a 3.9–4.3x gain over random baselines) "
            "while retaining authentic interactions at $\\mathrm{AP}_{\\mathrm{auth}} = 0.9989$."
        ),
        "future_work_focus": (
            "Integration of self-hosted, 4-bit quantized open-source local LLMs (such as LLaMA-3-8B) to eliminate external API costs, and extending "
            "the 2D BLAS burst detector to real-time streaming event architectures via Apache Kafka and Apache Flink."
        )
    },
    {
        "id": "vivaash",
        "name": "VIVAASH KOUNDAL",
        "reg_no": "23BCE5095",
        "file_prefix": "project_report_Vivaash_Koundal_23BCE5095",
        "pillar_title": "Risk-Aware Graph Contrastive Learning (GCL), Denoised SUPER Recommender Engine & Calibrated Pareto Debiasing",
        "short_pillar": "Risk-Aware GCL & Calibrated SUPER Debiasing",
        "role_summary": "Lead Graph Contrastive Representation & Calibrated Recommender Engine Architect",
        "modules_owned": [
            "models/simgcl.py",
            "models/sgl.py",
            "models/lightgcn.py",
            "trainer/contrastive_loss.py",
            "trainer/dual_trainer.py",
            "catalog/pareto_partition.py",
            "super_engine/blueprint.py",
            "super_engine/inclination.py",
            "super_engine/merger.py",
            "evaluation/ab_simulator.py"
        ],
        "key_responsibilities": (
            "Design and implementation of sample-reliability weighted InfoNCE contrastive alignment across SimGCL (latent uniform noise) and SGL "
            "(structural edge dropout), dual partition concurrent training of Head ($\mathcal{M}_{\\mathrm{pop}}$) and Tail ($\mathcal{M}_{\\mathrm{tail}}$) neural backbones, purified SUPER "
            "historical blueprint extraction ($w_{ui} \\ge \\theta_{\\mathrm{trust}}$), personalized tail inclination ($\\alpha_u$) calculation, rank-calibrated trajectory interleaving, "
            "formulation of attack-aware metrics ($g\\text{-}\\mathrm{LTC}@K$ and $\\mathrm{STER}@K$), and 1,000-sample Monte Carlo economic simulation."
        ),
        "teammates": [
            {
                "name": "ADVAIT RAVINDRAKUMAR VARHADE",
                "reg_no": "23BCE5098",
                "pillar": "Mathematical Noise-Transition Modeling, Tikhonov Operator Inversion & Risk-Consistent Proofs"
            },
            {
                "name": "P. MANAS",
                "reg_no": "23BCE5076",
                "pillar": "Multi-View Interaction Reliability Filter (MIRF), LLM Review Auditing & 2D BLAS Review-Bombing Detection"
            }
        ],
        "abstract_focus": (
            "With primary focus on graph contrastive representation learning and calibrated long-tail exposure, this report details how sample reliability "
            "weights modulate InfoNCE contrastive alignment losses and purify historical blueprints for personalized tail inclination ($\\alpha_u$) allocation. "
            "Empirical results prove statistically significant reductions in popularity calibration errors ($\\mathrm{RMSE\text{-}PC} = 0.1067, p = 0.0166$; $\\mathrm{MRMC} = 0.1128, p = 0.0088$), "
            "substantial suppression of spam target exposure ($\\mathrm{STER}@10$ reduced from $88\\%$ to $72\\%$), elevation of genuine long-tail catalog coverage ($g\\text{-}\\mathrm{LTC}@10 = 0.4202, p = 0.0397$ on Yelp), "
            "and a simulated annual GMV lift of USD 534,491 in offline Monte Carlo economic modeling."
        ),
        "future_work_focus": (
            "Distributed billion-edge graph partitioning using PyTorch-Geometric / DGL with CUDA IPC, and adapting the calibrated SUPER interleaving "
            "engine to decentralized federated graph recommendation settings (Fed-RRFN)."
        )
    }
]
