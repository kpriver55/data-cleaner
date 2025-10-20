"""
Optimization Page

Train and optimize the data cleaning agent using DSPy optimizers.
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Import optimization components
from optimization import (
    CleaningExample,
    CleaningDataset,
    OptimizationConfig,
    create_quick_optimization_config,
    create_balanced_optimization_config,
    create_thorough_optimization_config
)
from data_cleaning_agent import DataCleaningAgent
from file_io_tool import FileIOTool
from stats_tool import StatisticalAnalysisTool
from data_transformation_tool import DataTransformationTool
from utils import initialize_session_state

# Configure page
st.set_page_config(
    page_title="Optimization",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# Add optimization-specific session state
if 'optimization_dataset' not in st.session_state:
    st.session_state.optimization_dataset = None
if 'optimization_config' not in st.session_state:
    st.session_state.optimization_config = None
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None
if 'optimized_agent' not in st.session_state:
    st.session_state.optimized_agent = None

st.title("🎯 Model Optimization")
st.markdown("Train the AI on your specific cleaning tasks to improve performance")

# Create tabs for different stages
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Dataset Manager",
    "⚙️ Configuration",
    "🚀 Run Optimization",
    "📊 Evaluation"
])

# ==================== TAB 1: Dataset Manager ====================
with tab1:
    st.header("Training Dataset Manager")

    st.markdown("""
    Create and manage training examples for optimization. Each example should demonstrate
    a cleaning task with expected results.

    **Specification Options:**
    - **Option A (Best)**: Provide full cleaning plan text and rationale
    - **Option B (Good)**: Provide structured operations list
    """)

    # Upload existing dataset or create new
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Load Existing Dataset")
        uploaded_dataset = st.file_uploader(
            "Upload training dataset (JSON/YAML)",
            type=['json', 'yaml', 'yml'],
            help="Load an existing training dataset"
        )

        if uploaded_dataset is not None:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_dataset.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_dataset.read())
                    tmp_path = tmp_file.name

                # Load dataset
                if tmp_path.endswith('.json'):
                    dataset = CleaningDataset.from_json(tmp_path)
                else:
                    dataset = CleaningDataset.from_yaml(tmp_path)

                st.session_state.optimization_dataset = dataset
                st.success(f"✅ Loaded {len(dataset)} examples")

                # Show summary
                summary = dataset.summary()
                st.json(summary)

                # Clean up
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    with col2:
        st.subheader("Create New Dataset")
        if st.button("Create Empty Dataset"):
            st.session_state.optimization_dataset = CleaningDataset()
            st.success("✅ Created empty dataset")

    # Display and edit current dataset
    if st.session_state.optimization_dataset is not None:
        st.divider()
        st.subheader(f"Current Dataset ({len(st.session_state.optimization_dataset)} examples)")

        # Validate button
        if st.button("🔍 Validate Dataset"):
            errors = st.session_state.optimization_dataset.validate()
            if errors:
                st.error(f"Validation failed with {len(errors)} errors:")
                for error in errors:
                    st.write(f"- {error}")
            else:
                st.success("✅ Dataset is valid!")

        # Add new example form
        with st.expander("➕ Add New Example"):
            st.markdown("**Option A: Full Plan Specification** (Recommended for best quality)")

            input_file = st.file_uploader(
                "Upload messy dataset",
                type=['csv', 'xlsx', 'xls'],
                key="new_example_input"
            )

            example_description = st.text_input(
                "Description",
                placeholder="E.g., Clean customer data with duplicates and missing values"
            )

            option_a = st.checkbox("Use Option A (Full Plan)")
            option_b = st.checkbox("Use Option B (Structured Operations)")

            if option_a:
                expected_plan = st.text_area(
                    "Expected Cleaning Plan",
                    placeholder="""1. remove_duplicates(subset=['id'], keep='first')
2. handle_missing_values(columns=['age', 'salary'], numeric_strategy='median')
3. clean_text_columns(columns=['email'], operations=['strip', 'lower'])""",
                    height=150
                )

                expected_rationale = st.text_area(
                    "Expected Rationale",
                    placeholder="Explain why each operation is necessary and the order...",
                    height=100
                )

            if option_b:
                st.markdown("Define operations as JSON:")
                operations_json = st.text_area(
                    "Expected Operations (JSON)",
                    placeholder="""[
  {
    "operation": "remove_duplicates",
    "subset": ["id", "email"],
    "keep": "first",
    "rationale": "Remove duplicate records"
  },
  {
    "operation": "handle_missing_values",
    "columns": ["age", "salary"],
    "numeric_strategy": "median",
    "rationale": "Impute missing values"
  }
]""",
                    height=200
                )

            if st.button("Add Example"):
                if input_file is None:
                    st.error("Please upload an input file")
                elif not option_a and not option_b:
                    st.error("Please select Option A or Option B")
                else:
                    try:
                        # Save input file
                        input_dir = Path("training_data/inputs")
                        input_dir.mkdir(parents=True, exist_ok=True)
                        input_path = input_dir / input_file.name

                        with open(input_path, 'wb') as f:
                            f.write(input_file.read())

                        # Create example
                        example_data = {
                            'input_path': str(input_path),
                            'description': example_description if example_description else None
                        }

                        if option_a:
                            example_data['expected_cleaning_plan'] = expected_plan
                            example_data['expected_rationale'] = expected_rationale

                        if option_b:
                            try:
                                operations = json.loads(operations_json)
                                example_data['expected_operations'] = operations
                            except json.JSONDecodeError as e:
                                st.error(f"Invalid JSON for operations: {e}")
                                st.stop()

                        example = CleaningExample(**example_data)
                        st.session_state.optimization_dataset.add_example(example)

                        st.success(f"✅ Added example! Dataset now has {len(st.session_state.optimization_dataset)} examples")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error adding example: {e}")

        # Display examples
        if len(st.session_state.optimization_dataset) > 0:
            st.subheader("Examples")
            for i, example in enumerate(st.session_state.optimization_dataset):
                with st.expander(f"Example {i+1}: {example.description or example.input_path}"):
                    st.write(f"**Input:** {example.input_path}")
                    st.write(f"**Type:** {example.get_specification_type()}")

                    if example.expected_cleaning_plan:
                        st.write("**Expected Plan:**")
                        st.code(example.expected_cleaning_plan)
                        st.write("**Rationale:**")
                        st.write(example.expected_rationale)

                    if example.expected_operations:
                        st.write("**Expected Operations:**")
                        st.json(example.expected_operations)

        # Download dataset
        st.divider()
        st.subheader("💾 Save Dataset")

        col1, col2 = st.columns(2)

        with col1:
            dataset_name = st.text_input(
                "Dataset name",
                value=f"training_dataset_{datetime.now().strftime('%Y%m%d')}"
            )

        with col2:
            format_choice = st.selectbox("Format", ["JSON", "YAML"])

        if st.button("Save Dataset"):
            try:
                save_dir = Path("training_data/datasets")
                save_dir.mkdir(parents=True, exist_ok=True)

                if format_choice == "JSON":
                    save_path = save_dir / f"{dataset_name}.json"
                    st.session_state.optimization_dataset.to_json(save_path)
                else:
                    save_path = save_dir / f"{dataset_name}.yaml"
                    st.session_state.optimization_dataset.to_yaml(save_path)

                st.success(f"✅ Dataset saved to {save_path}")

                # Offer download
                with open(save_path, 'r') as f:
                    file_content = f.read()

                st.download_button(
                    label="📥 Download Dataset",
                    data=file_content,
                    file_name=save_path.name,
                    mime="application/json" if format_choice == "JSON" else "text/yaml"
                )

            except Exception as e:
                st.error(f"Error saving dataset: {e}")

# ==================== TAB 2: Configuration ====================
with tab2:
    st.header("Optimization Configuration")

    if st.session_state.optimization_dataset is None or len(st.session_state.optimization_dataset) == 0:
        st.warning("⚠️ Please create or load a training dataset first (Dataset Manager tab)")
    else:
        st.success(f"✅ Using dataset with {len(st.session_state.optimization_dataset)} examples")

        # Preset or custom configuration
        config_type = st.radio(
            "Configuration Type",
            ["Quick (Fast, few examples)", "Balanced (Recommended)", "Thorough (Best quality, slower)", "Custom"],
            help="Choose a preset or create custom configuration"
        )

        st.divider()

        # Configuration name
        config_name = st.text_input(
            "Configuration Name",
            value=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Unique name for this optimization run"
        )

        # Dataset path (temporary - will be saved)
        dataset_path_input = st.text_input(
            "Dataset Path",
            value="training_data/datasets/current_dataset.json",
            help="Path where dataset will be saved for optimization"
        )

        if config_type == "Custom":
            st.subheader("Optimizer Settings")

            optimizer_type = st.selectbox(
                "Optimizer Type",
                ["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPRO"],
                help="Choose the DSPy optimizer to use"
            )

            col1, col2 = st.columns(2)

            with col1:
                max_bootstrapped_demos = st.slider(
                    "Max Bootstrapped Demos",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="Maximum number of demonstrations to bootstrap"
                )

                max_labeled_demos = st.slider(
                    "Max Labeled Demos",
                    min_value=1,
                    max_value=20,
                    value=12,
                    help="Maximum number of labeled demonstrations to use"
                )

            with col2:
                num_candidate_programs = st.slider(
                    "Candidate Programs",
                    min_value=5,
                    max_value=30,
                    value=10,
                    help="Number of candidate programs to generate"
                )

                num_threads = st.slider(
                    "Threads",
                    min_value=1,
                    max_value=16,
                    value=8,
                    help="Number of parallel threads for optimization"
                )

            train_ratio = st.slider(
                "Train/Validation Split",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Ratio of data to use for training (rest for validation)"
            )

            # Create custom config
            from optimization import OptimizerConfig, EvaluationConfig

            optimizer_config = OptimizerConfig(
                optimizer_type=optimizer_type,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                num_candidate_programs=num_candidate_programs,
                num_threads=num_threads
            )

            config = OptimizationConfig(
                name=config_name,
                dataset_path=dataset_path_input,
                optimizer=optimizer_config,
                train_ratio=train_ratio
            )

        else:
            # Use preset
            preset_map = {
                "Quick (Fast, few examples)": create_quick_optimization_config,
                "Balanced (Recommended)": create_balanced_optimization_config,
                "Thorough (Best quality, slower)": create_thorough_optimization_config
            }

            config = preset_map[config_type](
                name=config_name,
                dataset_path=dataset_path_input
            )

        # Display configuration
        st.divider()
        st.subheader("Configuration Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Optimizer", config.optimizer.optimizer_type)
            st.metric("Max Demos", config.optimizer.max_bootstrapped_demos)

        with col2:
            st.metric("Candidate Programs", config.optimizer.num_candidate_programs)
            st.metric("Threads", config.optimizer.num_threads)

        with col3:
            st.metric("Train Ratio", f"{config.train_ratio:.0%}")
            st.metric("Examples", len(st.session_state.optimization_dataset))

        # Save configuration
        st.divider()
        if st.button("💾 Save Configuration"):
            try:
                # First save the dataset
                dataset_path = Path(dataset_path_input)
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                st.session_state.optimization_dataset.to_json(dataset_path)

                # Then save the config
                config_dir = Path("configs")
                config_dir.mkdir(exist_ok=True)
                config_path = config_dir / f"{config_name}.json"
                config.save_json(str(config_path))

                st.session_state.optimization_config = config

                st.success(f"✅ Configuration saved to {config_path}")
                st.success(f"✅ Dataset saved to {dataset_path}")

            except Exception as e:
                st.error(f"Error saving configuration: {e}")

# ==================== TAB 3: Run Optimization ====================
with tab3:
    st.header("Run Optimization")

    if st.session_state.optimization_config is None:
        st.warning("⚠️ Please create and save a configuration first (Configuration tab)")
    else:
        config = st.session_state.optimization_config

        st.success(f"✅ Configuration loaded: {config.name}")

        # Display configuration
        with st.expander("Configuration Details"):
            st.json(config.to_dict())

        # Check if agent is initialized
        if st.session_state.cleaning_agent is None:
            st.warning("⚠️ Please initialize the AI agent first")
            st.info("Go to the Data Cleaning page and configure the LLM")
        else:
            st.success("✅ AI agent is ready")

            # LLM configuration info
            if st.session_state.llm_config and st.session_state.llm_config.lm:
                llm_info = st.session_state.llm_config.get_current_config()
                st.info(f"📡 Using: {llm_info['provider']} / {llm_info['model']}")

            st.divider()

            # Optimization button
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("""
                **What will happen:**
                1. Load and validate training dataset
                2. Split into train/validation sets
                3. Run DSPy optimizer on the planning module
                4. Evaluate on train and validation sets
                5. Save optimized model

                ⏱️ **Estimated time:** 5-30 minutes depending on dataset size and optimizer
                """)

            with col2:
                run_optimization = st.button("🚀 Start Optimization", type="primary")

            if run_optimization:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                try:
                    with st.spinner("Running optimization..."):
                        # Run optimization
                        result = st.session_state.cleaning_agent.compile_with_optimizer(
                            config,
                            verbose=False  # We'll show our own progress
                        )

                        st.session_state.optimization_result = result
                        st.session_state.optimized_agent = st.session_state.cleaning_agent

                    st.success("🎉 Optimization completed!")

                    # Display results
                    st.subheader("Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Training Score",
                            f"{result.train_score:.4f}",
                            help="Average score on training examples"
                        )

                    with col2:
                        if result.val_score is not None:
                            st.metric(
                                "Validation Score",
                                f"{result.val_score:.4f}",
                                delta=f"{result.val_score - result.train_score:.4f}",
                                help="Score on held-out validation examples"
                            )
                        else:
                            st.metric("Validation Score", "N/A")

                    with col3:
                        st.metric(
                            "Duration",
                            f"{result.duration_seconds:.1f}s",
                            help="Time taken for optimization"
                        )

                    # Full results
                    with st.expander("Detailed Results"):
                        st.write(result.summary())

                    # Save model
                    st.divider()
                    st.subheader("💾 Save Optimized Model")

                    model_name = st.text_input(
                        "Model name",
                        value=f"optimized_{config.name}"
                    )

                    if st.button("Save Model"):
                        try:
                            model_dir = Path("models")
                            model_dir.mkdir(exist_ok=True)
                            model_path = model_dir / f"{model_name}.pkl"

                            st.session_state.cleaning_agent.save_compiled_model(
                                str(model_path),
                                metadata={
                                    'config': config.to_dict(),
                                    'result': result.to_dict()
                                }
                            )

                            st.success(f"✅ Model saved to {model_path}")

                        except Exception as e:
                            st.error(f"Error saving model: {e}")

                except Exception as e:
                    st.error(f"❌ Optimization failed: {e}")
                    st.exception(e)

# ==================== TAB 4: Evaluation ====================
with tab4:
    st.header("Model Evaluation")

    st.markdown("""
    Evaluate optimized models on test datasets to measure performance improvements.
    """)

    # Load model
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Load Model")

        model_file = st.file_uploader(
            "Upload model (.pkl)",
            type=['pkl'],
            help="Load a saved optimized model"
        )

        if model_file is not None:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(model_file.read())
                    tmp_path = tmp_file.name

                # Load model
                io_tool = FileIOTool()
                stats_tool = StatisticalAnalysisTool()
                transform_tool = DataTransformationTool()

                loaded_agent = DataCleaningAgent.load_compiled_model(
                    tmp_path,
                    io_tool,
                    stats_tool,
                    transform_tool
                )

                st.session_state.optimized_agent = loaded_agent
                st.success("✅ Model loaded successfully")

                # Clean up
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Error loading model: {e}")

    with col2:
        st.subheader("Load Test Dataset")

        test_dataset_file = st.file_uploader(
            "Upload test dataset (JSON/YAML)",
            type=['json', 'yaml', 'yml'],
            help="Load a test dataset for evaluation"
        )

        if test_dataset_file is not None:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(test_dataset_file.name).suffix) as tmp_file:
                    tmp_file.write(test_dataset_file.read())
                    tmp_path = tmp_file.name

                # Load dataset
                if tmp_path.endswith('.json'):
                    test_dataset = CleaningDataset.from_json(tmp_path)
                else:
                    test_dataset = CleaningDataset.from_yaml(tmp_path)

                st.session_state.test_dataset = test_dataset
                st.success(f"✅ Loaded {len(test_dataset)} test examples")

                # Clean up
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Error loading test dataset: {e}")

    # Run evaluation
    if st.session_state.get('optimized_agent') and st.session_state.get('test_dataset'):
        st.divider()

        if st.button("📊 Run Evaluation"):
            try:
                with st.spinner("Evaluating model..."):
                    results = st.session_state.optimized_agent.evaluate_on_dataset(
                        st.session_state.test_dataset,
                        verbose=False
                    )

                st.success("✅ Evaluation complete!")

                # Display results
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Average Score", f"{results['average_score']:.4f}")

                with col2:
                    st.metric("Min Score", f"{results['min_score']:.4f}")

                with col3:
                    st.metric("Max Score", f"{results['max_score']:.4f}")

                with col4:
                    st.metric("Examples", results['num_examples'])

                # Per-example results
                st.subheader("Per-Example Results")

                details_df = pd.DataFrame(results['details'])
                st.dataframe(details_df, use_container_width=True)

                # Download results
                st.divider()
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=results_json,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                st.exception(e)

    elif not st.session_state.get('optimized_agent'):
        st.info("Please load a model first")
    elif not st.session_state.get('test_dataset'):
        st.info("Please load a test dataset first")
