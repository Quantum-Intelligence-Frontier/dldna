import torch

class TaskSpecificMasking:
    @staticmethod
    def create_pad_mask(seq, pad_idx=0):
        """Create padding mask"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    @staticmethod
    def create_subsequent_mask(size):
        """Create causality mask"""
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0
    
    @staticmethod
    def create_mlm_mask(seq, mask_prob=0.15, mask_token_id=103):
        """Create mask for MLM task"""
        mask = torch.rand(seq.shape) < mask_prob
        return mask, mask_token_id
    
    @staticmethod
    def get_mask(task_type, **kwargs):
        """Create task-specific mask"""
        if task_type == 'copy':
            seq = kwargs.get('seq')
            pad_idx = kwargs.get('pad_idx', 0)
            return TaskSpecificMasking.create_pad_mask(seq, pad_idx)
            
        elif task_type == 'causal':
            size = kwargs.get('size')
            return TaskSpecificMasking.create_subsequent_mask(size)
            
        elif task_type == 'mlm':
            seq = kwargs.get('seq')
            mask_prob = kwargs.get('mask_prob', 0.15)
            mask_token_id = kwargs.get('mask_token_id', 103)
            return TaskSpecificMasking.create_mlm_mask(seq, mask_prob, mask_token_id)
            
        elif task_type == 'addition':
            seq_length = kwargs.get('seq_length')
            valid_length = kwargs.get('valid_length')
            return TaskSpecificMasking.create_padding_mask(seq_length, valid_length) # Assuming a create_padding_mask function exists.
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")