from .segmentors import DeepLabV3Plus, get_maskrcnn_model

def build_model(config):
    """
    Model Factory
    """
    task_type = config.get('task_type', 'semantic')
    
    if task_type == 'semantic':
        return DeepLabV3Plus(
            num_classes=config['model']['num_classes'],
            backbone_name=config['model']['backbone'],
            pretrained=config['model']['pretrained'],
            output_stride=config['model']['output_stride'],
            use_cbam=config['model']['use_cbam']
        )
    elif task_type == 'instance':
        # Mask R-CNN
        print("Building Instance Segmentation Model (Mask R-CNN)")
        # Get maskrcnn config, fallback to empty dict if not present (for safety)
        m_cfg = config['model'].get('maskrcnn', {})
        return get_maskrcnn_model(
            num_classes=config['model']['num_classes'],
            pretrained=m_cfg.get('pretrained', True),
            min_size=m_cfg.get('min_size', 512),
            max_size=m_cfg.get('max_size', 800),
            trainable_backbone_layers=m_cfg.get('trainable_backbone_layers', 3)
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
