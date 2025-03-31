from setuptools import setup, find_packages

setup(
    name="sp500_forecasting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
        "joblib>=1.1.0",
        "statsmodels>=0.13.0",
        "tqdm>=4.62.0",
    ],
    author="Alex Garnyk",
    author_email="garnykalex@gmail.com",
    description="S&P 500 forecasting system using machine learning",
    keywords="finance, machine learning, forecasting, stock market, S&P 500",
    url="https://github.com/yourusername/sp500_forecasting_project",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)