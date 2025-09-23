from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ultrathink-pipeline",
    version="1.0.0",
    description="ULTRATHINK: Advanced AI Training Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ULTRATHINK Team",
    author_email="team@ultrathink.ai",
    url="https://github.com/yourusername/ultrathink",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, machine learning, transformers, llm, training, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ultrathink/issues",
        "Source": "https://github.com/yourusername/ultrathink",
        "Documentation": "https://github.com/yourusername/ultrathink/blob/main/PROJECT_OVERVIEW.md",
    },
    entry_points={
        "console_scripts": [
            "ultrathink-train=train_ultrathink:main",
            "ultrathink-infer=scripts.inference:main",
        ],
    },
)
