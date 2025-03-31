import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
import time

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def execute_script(script_path, description, env=None):
    """
    Execute script with error handling and return results

    Args:
        script_path: Path to Python script
        description: Description of operation for logging
        env: Additional environment variables (dictionary)

    Returns:
        Tuple: (success, output, error, elapsed_time)
    """
    logger.info(f"Executing: {description} ({script_path})")
    print(f"\n[*] {description}...")

    start_time = time.time()

    try:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Start process
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=process_env
        )

        # Get output
        stdout, stderr = process.communicate()

        elapsed_time = time.time() - start_time

        # Log output
        if stdout:
            for line in stdout.splitlines():
                logger.info(f"  {line}")

        # Check errors
        if process.returncode != 0:
            logger.error(f"Script failed with exit code: {process.returncode}")
            if stderr:
                for line in stderr.splitlines():
                    logger.error(f"  {line}")

            print(
                f"[!] Failed ({elapsed_time:.2f}s): {stderr.splitlines()[-1] if stderr.splitlines() else 'Unknown error'}")
            return False, stdout, stderr, elapsed_time

        print(f"[+] Completed ({elapsed_time:.2f}s)")
        return True, stdout, stderr, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Unexpected error: {str(e)}")
        print(f"[!] Error ({elapsed_time:.2f}s): {str(e)}")
        return False, "", str(e), elapsed_time


def check_requirements():
    """Check necessary requirements for execution"""
    try:
        # Check for necessary directories
        for directory in ['data', 'models', 'reports']:
            os.makedirs(directory, exist_ok=True)

        # Check Python packages with correct import names
        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',  # Important fix: scikit-learn is imported as sklearn
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'joblib': 'joblib',
            'sqlite3': 'sqlite3'
        }

        missing_packages = []
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                if package_name != 'sqlite3':  # sqlite3 is usually included in Python
                    missing_packages.append(package_name)

        if missing_packages:
            logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
            return False, f"Missing packages: {', '.join(missing_packages)}"

        return True, "All requirements satisfied"

    except Exception as e:
        logger.error(f"Error checking requirements: {str(e)}")
        return False, str(e)


def run_pipeline(args):
    """
    Run complete data processing and forecasting pipeline

    Args:
        args: Command line arguments with parameters
    """
    # Check requirements
    requirements_met, message = check_requirements()
    if not requirements_met:
        logger.error(f"Requirements check failed: {message}")
        print(f"[!] Requirements not met: {message}")
        print(f"    Please install missing dependencies with pip before running the pipeline.")
        return False

    logger.info("=== Starting S&P 500 forecast pipeline ===")
    logger.info(f"Forecast period: {args.days} days")
    logger.info(f"Process mode: {args.mode}")

    print("=== S&P 500 FORECASTING PIPELINE ===")
    print(f"Target: {args.days}-day forecast")

    # Define script files based on selected mode
    if args.mode == 'full':
        scripts = [
            ("01_data_fetch.py", "Fetching historical S&P 500 data"),
            ("02_data_preprocess.py", "Preprocessing data and creating features"),
            ("03_model_train.py", "Training predictive model"),
            ("05_forecast.py", "Generating S&P 500 forecast")
        ]
    elif args.mode == 'update':
        scripts = [
            ("01_data_fetch.py", "Updating S&P 500 data"),
            ("02_data_preprocess.py", "Updating features"),
            ("05_forecast.py", "Generating updated forecast")
        ]
    elif args.mode == 'forecast':
        # Only forecasting without updating data and retraining
        scripts = [
            ("05_forecast.py", "Generating forecast with existing model")
        ]
    else:
        logger.error(f"Unknown mode: {args.mode}")
        print(f"[!] Invalid mode: {args.mode}")
        return False

    # Execute scripts sequentially
    all_success = True

    for script, description in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)

        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            print(f"[!] Missing script: {script}")
            all_success = False
            break

        # Prepare environment variables for current script
        env = {}

        # Configure options for specific scripts
        if script == "05_forecast.py":
            env["FORECAST_DAYS"] = str(args.days)
        elif script == "03_model_train.py" and args.target:
            env["TARGET_TYPE"] = args.target

        # Run script
        success, output, error, elapsed_time = execute_script(script_path, description, env)

        if not success:
            all_success = False
            break

    # Final summary
    if all_success:
        logger.info("=== Pipeline completed successfully ===")
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Forecast results are available in reports/ directory")
        print("- Check reports/forecast_report.txt for detailed summary")
        print("- View visualizations in reports/ folder")
        return True
    else:
        logger.error("=== Pipeline failed ===")
        print("\n=== PIPELINE FAILED ===")
        print("Check the log file for details")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='S&P 500 Forecasting Pipeline')

    parser.add_argument('--days', type=int, default=90,
                        help='Number of days to forecast (default: 90)')

    parser.add_argument('--mode', choices=['full', 'update', 'forecast'], default='full',
                        help='Pipeline mode: full (all steps), update (fetch data & forecast), forecast (only)')

    parser.add_argument('--target', choices=['next_price', 'next_return', 'log_price'],
                        help='Target variable type for model training')

    args = parser.parse_args()

    # Run pipeline
    success = run_pipeline(args)

    # Return code
    sys.exit(0 if success else 1)