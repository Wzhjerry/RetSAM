from models.swin_multitask import Swin_Multitask
from models.swin_multitask_final import Swin_Multitask_Final


def init_model(args):

    # Only support multitask variants in this codebase
    if args.model == 'swin_multitask':
        model = Swin_Multitask(
            in_channels=3,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            window_size=args.window_size,
            depths=args.depths,
            num_heads=args.num_heads,
            feature_size=args.feature_size,
            use_v2=False,
        )
    elif args.model == 'swin_multitask_final':
        model = Swin_Multitask_Final(
            in_channels=3,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            window_size=args.window_size,
            depths=args.depths,
            num_heads=args.num_heads,
            feature_size=args.feature_size,
            use_v2=False,
            img_size=getattr(args, 'size', 640),
            # Coordinate prediction parameters
            enable_coordinate_prediction=getattr(args, 'enable_coordinate_prediction', False),
            num_coordinates=getattr(args, 'num_coordinates', 2),
            coordinate_hidden_size=getattr(args, 'coordinate_hidden_size', 512),
            coordinate_dropout=getattr(args, 'coordinate_dropout', 0.1),
            use_attention_head=getattr(args, 'use_attention_head', False),
        )
    else:
        raise NotImplementedError(f"Model '{args.model}' is not supported in this build (multitask only).")

    return model
