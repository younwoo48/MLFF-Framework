import torch

from ocpmodels.common.registry import registry

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper

from src.trainers.forces_trainer import ForcesTrainer
from src.common.collaters.parallel_collater_nequip import ParallelCollaterNequIP


@registry.register_trainer("forces_nequip")
class NequIPForcesTrainer(ForcesTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) task, 
    and this class is especially used to train an NequIP model.
    """
    def __init__(self, config):
        super().__init__(config)

        # copied from nequip/scripts/train.py
        if int(torch.__version__.split(".")[1]) >= 11:
            # PyTorch >= 1.11
            torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
        else:
            torch._C._jit_set_bailout_depth(2)
    
    def _parse_config(self, config):
        trainer_config = super()._parse_config(config)
        trainer_config["model_attributes"]["dataset"] = trainer_config["dataset"]
        return trainer_config
    
    # TODO: remove this function
    def _update_model_attributes_config(self):
        # this function is called in self._set_model()
        # self.config["model_attributes"]["additional_config"]["dataset"] = self.train_dataset
        self.config["model_attributes"]["dataset"] = self.train_dataset
    
    def initiate_collater(self):
        self.type_mapper = TypeMapper(
            type_names=self.config["model_attributes"].get("type_names", None),
            chemical_symbol_to_type=self.config["model_attributes"].get("chemical_symbol_to_type", None),
            chemical_symbols=self.config["model_attributes"].get("chemical_symbols", None),
        )
        return ParallelCollaterNequIP(
            num_gpus=0 if self.cpu else 1,
            otf_graph=self.config["model_attributes"].get("otf_graph", False),
            use_pbc=self.config["model_attributes"].get("use_pbc", False),
            type_mapper=self.type_mapper,
        )

    def _set_model(self):
        super()._set_model()
        
        # for yaml.dump, lmdb instance should be excluded from the configuration after model setup
        # self.config["model_attributes"].pop("dataset")

    def _compute_loss(self, out, batch_list):
        # loss function always needs to be in normalized unit (according to NequIP)
        if self.model.training:
            # loss used in train mode
            # target is converted real unit -> normalized unit 
            # (prediction is not touched, because it is in normalized unit)
            normalized_batch_list = []
            for batch in batch_list:
                b = self._unwrapped_model.do_unscale(
                    data=AtomicData.to_AtomicDataDict(batch)
                )
                normalized_batch = AtomicData.from_AtomicDataDict(b)
                normalized_batch.y = b[AtomicDataDict.TOTAL_ENERGY_KEY]
                normalized_batch.force = b[AtomicDataDict.FORCE_KEY]
                normalized_batch.natoms = torch.bincount(batch[AtomicDataDict.BATCH_KEY])
                normalized_batch_list.append(normalized_batch)
            return super()._compute_loss(out=out, batch_list=normalized_batch_list)
        else:
            # loss used in eval mode
            with torch.no_grad():
                # target is converted real unit -> normalized unit
                normalized_batch_list = []
                for batch in batch_list:
                    b = self._unwrapped_model.do_unscale(
                        data=AtomicData.to_AtomicDataDict(batch),
                        force_process=True,
                    )
                    normalized_batch = AtomicData.from_AtomicDataDict(b)
                    normalized_batch.y = b[AtomicDataDict.TOTAL_ENERGY_KEY]
                    normalized_batch.force = b[AtomicDataDict.FORCE_KEY]
                    normalized_batch.natoms = torch.bincount(batch[AtomicDataDict.BATCH_KEY])
                    normalized_batch_list.append(normalized_batch)
                # prediction is converted ? -> normalized unit
                _out = self._unwrapped_model.do_unscale(
                    {
                        AtomicDataDict.TOTAL_ENERGY_KEY: out["energy"], 
                        AtomicDataDict.FORCE_KEY: out["forces"],
                    }, 
                    force_process=True,
                )
                normalized_out = {
                    "energy": _out[AtomicDataDict.TOTAL_ENERGY_KEY],
                    "forces": _out[AtomicDataDict.FORCE_KEY],
                }
                # print("-> eval mode loss energy GT:", batch_list[0].y.tolist(), "Pred:", out["energy"].tolist())
                return super()._compute_loss(out=normalized_out, batch_list=normalized_batch_list)

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        # print("[before metrics] energy GT:", batch_list[0][AtomicDataDict.TOTAL_ENERGY_KEY].tolist(), "Pred:", out["energy"]).tolist()
        for batch in batch_list:
            batch.y = batch[AtomicDataDict.TOTAL_ENERGY_KEY]
            batch.force = batch[AtomicDataDict.FORCE_KEY]
            batch.natoms = torch.bincount(batch[AtomicDataDict.BATCH_KEY])

        if self.model.training:
            # train mode
            # prediction is converted normalized unit -> real unit
            with torch.no_grad():
                _out = self._unwrapped_model.undo_unscale(
                    {
                        AtomicDataDict.TOTAL_ENERGY_KEY: out["energy"], 
                        AtomicDataDict.FORCE_KEY: out["forces"],
                    },
                    force_process=True,
                )
                out["energy"] = _out[AtomicDataDict.TOTAL_ENERGY_KEY]
                out["forces"] = _out[AtomicDataDict.FORCE_KEY]
        
        return super()._compute_metrics(out=out, batch_list=batch_list, evaluator=evaluator, metrics=metrics)    