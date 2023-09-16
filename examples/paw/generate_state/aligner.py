import torch

class BypassSegmentor(object):
    def __init__(
        self,
        models,
        tgt_dict,
        text_seg_type="word",
        use_cuda=True,
        debug_level=False,
    ):
        super().__init__()
        """ Calculate alignment (CA) with three (3) information; char, speech and text 
        """
        # Model related
        self.model = models
        self.tgt_dict = tgt_dict

        # self.bos = self.tgt_dict.bos()
        self.pad = self.tgt_dict.pad()
        self.blank = (
            self.tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in self.tgt_dict.indices
            else self.tgt_dict.bos()
        )
        self.split = self.tgt_dict.unk() + 1
        self.use_cuda = use_cuda

        # Data related
        self.text_seg_type = text_seg_type

        # Other
        self.debug_level = debug_level

    def _generate_pl_and_duration(
        self,
        sample,
        prop2raw=False
    ):
        # Generate emissions of the s2t model
        net_output = self.model(**sample["net_input"], mask=False)
        emissions = self.model.get_normalized_probs(net_output, log_probs=False)
        emissions = emissions.transpose(1, 0) # [tsz, bsz, chsz] -> [bsz, tsz, chsz]

        results = {
            "state": emissions.argmax(-1).detach().cpu().numpy(),
            "ltr" : None,
            "dur" : None
        }

        return results 
    
    def segment(
        self,
        sample,
        prop2raw=False
    ):
        res = self._generate_pl_and_duration(sample, prop2raw)

        return res