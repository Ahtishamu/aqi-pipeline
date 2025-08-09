#!/usr/bin/env python3
"""
Update Hopsworks Model Registry with trained models
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import joblib
import hopsworks
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_training_metadata(model_dir_path: Path):
    info_files = list(model_dir_path.glob("training_info_*.json"))
    if not info_files:
        return {}
    latest = max(info_files, key=lambda x: x.stat().st_mtime)
    try:
        with open(latest, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load training metadata {latest}: {e}")
        return {}


def _build_model_schema(feature_columns, target_name):
    try:
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        if not feature_columns:
            return None
        input_schema = Schema(feature_columns)
        output_schema = Schema([target_name])
        return ModelSchema(input_schema=input_schema, output_schema=output_schema)
    except Exception as e:
        logger.warning(f"Could not build model schema (will proceed without it): {e}")
        return None


def _create_and_save_model(mr, framework: str, name: str, description: str, model_object_path: Path,
                           model_object, model_schema, metrics: dict, input_example: dict):
    """Try various HSML API creation patterns until one works."""
    last_error = None

    # Candidate creation call specs (method, kwargs, save_uses_object)
    attempts = []
    if hasattr(mr, 'python') and hasattr(mr.python, 'create_model'):
        attempts.append(('mr.python.create_model', mr.python.create_model, [
            {"name": name, "description": description, "model_schema": model_schema, "input_example": input_example, "metrics": metrics},
            {"name": name, "description": description, "model_schema": model_schema, "input_example": input_example},
            {"name": name, "description": description}
        ], 'object'))
    if hasattr(mr, 'sklearn') and hasattr(mr.sklearn, 'create_model'):
        attempts.append(('mr.sklearn.create_model', mr.sklearn.create_model, [
            {"name": name, "description": description, "metrics": metrics, "model": model_object, "model_schema": model_schema, "input_example": input_example},
            {"name": name, "description": description, "model": model_object},
            {"name": name, "description": description}
        ], 'implicit'))

    # Some environments expose a generic create / get or register method on model objects
    if hasattr(mr, 'create_model'):
        attempts.append(('mr.create_model', mr.create_model, [
            {"name": name, "description": description},
            {"name": name}
        ], 'generic'))

    for label, func, kw_variants, mode in attempts:
        for kwargs in kw_variants:
            try:
                logger.info(f"Trying model registration via {label} with args: {list(kwargs.keys())}")
                model_handle = func(**kwargs)

                # Saving artifact
                try:
                    if hasattr(model_handle, 'save'):
                        # Prefer saving object if available
                        if mode in ('object', 'implicit') and model_object is not None:
                            try:
                                model_handle.save(model_object, model_schema=model_schema)
                            except TypeError:
                                # Fallback: maybe expects path
                                model_handle.save(str(model_object_path))
                        else:
                            model_handle.save(str(model_object_path))
                    else:
                        # As a last resort, if model_handle has upload / add functions
                        if hasattr(model_handle, 'upload'):
                            model_handle.upload(str(model_object_path))
                except Exception as save_err:
                    logger.warning(f"Artifact save step encountered an issue (continuing): {save_err}")

                # Metrics update attempts
                if metrics:
                    updated = False
                    for attr in ('set_metrics', 'update_metrics', 'log_metrics'):
                        if hasattr(model_handle, attr):
                            try:
                                getattr(model_handle, attr)(metrics)
                                updated = True
                                break
                            except Exception as m_err:
                                logger.warning(f"Metric method {attr} failed: {m_err}")
                    if not updated:
                        try:
                            setattr(model_handle, 'metrics', metrics)
                        except Exception:
                            pass
                logger.info("Model registration succeeded")
                return True
            except Exception as e:
                last_error = e
                logger.debug(f"Attempt with {label} kwargs {kwargs} failed: {e}")
    logger.error(f"All model registration attempts failed. Last error: {last_error}")
    return False


def update_model_registry(horizon: str, model_dir: str, best_model_name: str):
    """Update Hopsworks Model Registry with the trained model."""
    try:
        logger.info(f"📦 Updating model registry for {horizon}")

        api_key = os.environ.get('HOPSWORKS_API_KEY')
        if not api_key:
            logger.error("HOPSWORKS_API_KEY not found")
            return False

        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()

        model_dir_path = Path(model_dir)
        model_files = list(model_dir_path.glob("best_model_*.pkl"))
        results_files = list(model_dir_path.glob("model_results_*.csv"))
        if not model_files or not results_files:
            logger.error(f"Model files not found in {model_dir}")
            return False

        latest_results_file = max(results_files, key=lambda x: x.stat().st_mtime)
        results_df = pd.read_csv(latest_results_file)
        logger.info(f"Available models in results: {results_df['Model'].tolist()}")

        # Robust best model selection
        search_order = [
            lambda df: df[df['Model'].str.lower() == best_model_name.lower()],
            lambda df: df[df['Model'].str.replace('_', ' ', case=False).str.lower() == best_model_name.replace('_', ' ').lower()],
            lambda df: df[df['Model'].str.contains(best_model_name.replace('_', ' '), case=False, na=False)],
        ]
        best_row = None
        for strat in search_order:
            try:
                candidate = strat(results_df)
                if not candidate.empty:
                    best_row = candidate.iloc[0]
                    break
            except Exception:
                continue
        if best_row is None:
            best_row = results_df.iloc[0]
            logger.warning(f"Could not match model name '{best_model_name}', using first row: {best_row['Model']}")
        else:
            logger.info(f"Matched best model row: {best_row['Model']}")

        metrics = {
            'rmse': float(best_row.get('RMSE', best_row.get('rmse', 0))),
            'mae': float(best_row.get('MAE', best_row.get('mae', 0))),
            'r2_score': float(best_row.get('R2', best_row.get('r2', 0)))
        }

        # Locate actual model artifact for this horizon
        pattern = f"best_model_aqi_t_{horizon}_*.pkl"
        horizon_models = list(model_dir_path.glob(pattern))
        if not horizon_models:
            # fallback to any best_model file (already collected earlier)
            logger.warning(f"No horizon-specific model with pattern {pattern}, using latest generic model file")
            actual_model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        else:
            actual_model_path = max(horizon_models, key=lambda x: x.stat().st_mtime)

        # Load model object
        try:
            model_object = joblib.load(actual_model_path)
        except Exception as e:
            logger.error(f"Failed to load model pickle {actual_model_path}: {e}")
            return False

        training_meta = _load_training_metadata(model_dir_path)
        feature_columns = training_meta.get('feature_columns') or training_meta.get('features') or []
        if not feature_columns:
            # attempt to infer from model_results columns: exclude known metric columns
            exclude = {'Model', 'RMSE', 'MAE', 'R2', 'rmse', 'mae', 'r2', 'Timestamp'}
            feature_columns = [c for c in results_df.columns if c not in exclude]
        target_name = f"aqi_t_{horizon}".replace('h', '')  # nominal
        model_schema = _build_model_schema(feature_columns, target_name)
        # Minimal input example
        input_example = {col: 0 for col in feature_columns[:10]}  # limit size

        model_name = f"aqi_prediction_{horizon}"
        success = _create_and_save_model(
            mr=mr,
            framework='sklearn',
            name=model_name,
            description=f"AQI prediction model for {horizon} horizon using {best_model_name}",
            model_object_path=actual_model_path,
            model_object=model_object,
            model_schema=model_schema,
            metrics=metrics,
            input_example=input_example
        )
        if not success:
            return False

        logger.info(f"✅ Model {model_name} registered with metrics: {metrics}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to update model registry: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Update model registry')
    parser.add_argument('--horizon', choices=['24h', '48h', '72h'], required=True, help='Model horizon')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing model files')
    parser.add_argument('--best-model', type=str, required=True, help='Name of the best model')
    args = parser.parse_args()
    success = update_model_registry(args.horizon, args.model_dir, args.best_model)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
