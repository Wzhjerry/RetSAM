# Model factory and module imports
from models.model_factory import init_model
from models.base_module import Base_Module
from models.multitask_module import MultiTask_Module
from models.multitask_final_module import MultiTask_Final_Module


def load_model(args):
    """Factory function to load supported multitask modules"""
    
    if args.model in ['swin_multitask']:
        model = MultiTask_Module(args)
    elif args.model in ['swin_multitask_final']:
        model = MultiTask_Final_Module(args)
    else:
        raise NotImplementedError(f"Model '{args.model}' is not supported (multitask only).")
        
    return model


def test():
    """Test function for model initialization"""
    import argparse
    
    # Create a simple test args object
    args = argparse.Namespace()
    args.model = 'swin_multitask'
    args.size = 640
    args.out_channels = [2, 3, 2, 4, 6]  # Example multitask channels
    args.class_weights = [1.0] * len(args.out_channels)
    args.patch_size = 4
    args.window_size = 12
    args.depths = [2, 2, 18, 2]
    args.num_heads = [6, 12, 24, 48]
    args.feature_size = 48
    args.pretrained = None
    args.debug = True
    args.batch_size = 2
    
    # Test model initialization
    try:
        model = load_model(args)
        print(f"Successfully initialized {args.model}")
        print(f"Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"Failed to initialize {args.model}: {e}")
        return False


# Maintain backward compatibility - re-export key functions
__all__ = [
    'init_model',
    'load_model', 
    'Base_Module',
    'MultiTask_Module',
    'MultiTask_Final_Module',
    'test'
]