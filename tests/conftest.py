"""
Pytest configuration and fixtures for qtaim_embed tests.

Provides device detection, conditional test markers, and fixtures for
CPU, single GPU, and multi-GPU testing.
"""
import pytest
import torch


def get_available_devices():
    """
    Detect available compute devices.

    Returns:
        dict: Available devices with keys:
            - 'cpu': Always True
            - 'gpu': True if CUDA available
            - 'gpu_count': Number of GPUs (0 if none)
            - 'multi_gpu': True if 2+ GPUs available
    """
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    return {
        'cpu': True,
        'gpu': gpu_available,
        'gpu_count': gpu_count,
        'multi_gpu': gpu_count >= 2,
    }


def get_test_devices_config(prefer_single_gpu=True):
    """
    Get the appropriate device configuration for tests.

    For small test datasets, using a single GPU avoids BatchNorm issues
    with distributed training where batches can become too small per GPU.

    Args:
        prefer_single_gpu (bool): If True and multiple GPUs available,
            use only 1 GPU to avoid batch size issues in tests.

    Returns:
        dict: Device config with keys:
            - 'accelerator': 'auto', 'cpu', or 'gpu'
            - 'devices': Number of devices to use (1 for CPU, 1+ for GPU)
            - 'strategy': 'ddp' only if multi-GPU (omitted for single device)
    """
    devices = get_available_devices()

    if not devices['gpu']:
        # No GPU available - use CPU
        return {
            'accelerator': 'cpu',
            'devices': 1,
        }

    if prefer_single_gpu or devices['gpu_count'] == 1:
        # Single GPU preferred or only one available
        return {
            'accelerator': 'auto',
            'devices': 1,
        }

    # Multi-GPU available and requested
    return {
        'accelerator': 'auto',
        'devices': devices['gpu_count'],
        'strategy': 'ddp',
    }


# Pytest markers for conditional test execution
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring a GPU (skipped if no GPU available)"
    )
    config.addinivalue_line(
        "markers", "multi_gpu: mark test as requiring 2+ GPUs (skipped if fewer available)"
    )
    config.addinivalue_line(
        "markers", "cpu_only: mark test as CPU-only (never uses GPU)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available hardware."""
    devices = get_available_devices()

    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_multi_gpu = pytest.mark.skip(reason="Multiple GPUs not available")

    for item in items:
        if "gpu" in item.keywords and not devices['gpu']:
            item.add_marker(skip_gpu)
        if "multi_gpu" in item.keywords and not devices['multi_gpu']:
            item.add_marker(skip_multi_gpu)


# Fixtures for different device configurations
@pytest.fixture
def device_config():
    """
    Fixture providing appropriate device config for tests.
    Uses single GPU for small test datasets to avoid BatchNorm issues.
    """
    return get_test_devices_config(prefer_single_gpu=True)


@pytest.fixture
def multi_gpu_config():
    """
    Fixture providing multi-GPU config for distributed training tests.
    Only use this for tests with large enough batches.
    """
    devices = get_available_devices()
    if not devices['multi_gpu']:
        pytest.skip("Multiple GPUs not available")

    return {
        'accelerator': 'auto',
        'devices': devices['gpu_count'],
        'strategy': 'ddp',
    }


@pytest.fixture
def cpu_config():
    """Fixture forcing CPU-only execution."""
    return {
        'accelerator': 'cpu',
        'devices': 1,
    }


@pytest.fixture
def single_gpu_config():
    """Fixture for single GPU execution."""
    devices = get_available_devices()
    if not devices['gpu']:
        pytest.skip("GPU not available")

    return {
        'accelerator': 'auto',
        'devices': 1,
    }
