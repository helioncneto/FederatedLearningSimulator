from abc import ABC, abstractmethod
from global_update_method import base_aggregation, delta_aggregation_slowmo, adam_aggregation, delta_aggregation, \
    delta_aggregation_fedDyn, delta_aggregation_AGM, base_aggregation_fedSA, delta_aggregation_AGM_fedSA, \
    base_aggregation_IG, base_aggregation_fedSA_IG, delta_aggregation_AGM_fedSA_IG, base_aggregation_reputation, \
    base_aggregation_Oort, delta_aggregation_AGM_IG

from local_update_method import base, weight_l2, fedCM, fedDyn, fedAGM


class IGlobalMethodFactory(ABC):
    """Abstract class for global method classes"""

    @abstractmethod
    def get_global_method(self, args, device, trainset, testloader, valloader, local_update):
        """Return the global method class"""


class ILocalMethodFactory(ABC):
    """Abstract class for global method classes"""

    @abstractmethod
    def get_local_method(self):
        """Return the global method class"""


class GlobalBaseAggregationFactory(IGlobalMethodFactory):
    """Method for returning the FedAVG global aggregation method"""

    def get_global_method(self, args, device, trainset, testloader, valloader, local_update):
        return base_aggregation.BaseGlobalUpdate(args=args, device=device, trainset=trainset, testloader=testloader,
                                                 valloader=valloader, local_update=local_update)

class GlobalBaseAggregationIGFactory(IGlobalMethodFactory):
    """Method for returning the FedAVG using IG for selection"""

    def get_global_method(self,  args, device, trainset, testloader, valloader, local_update):
        return base_aggregation_IG.FedSBSGlobalUpdate(args=args, device=device, trainset=trainset, testloader=testloader,
                                                 valloader=valloader, local_update=local_update)


class GlobalDeltaAggregationSlowmoFactory(IGlobalMethodFactory):
    """Method for returning the Slowmo global aggregation method"""

    def get_global_method(self):
        return delta_aggregation_slowmo.GlobalUpdate


class GlobalAdamAggregationFactory(IGlobalMethodFactory):
    """Method for returning the FeAVG with Adam global aggregation method"""

    def get_global_method(self):
        return adam_aggregation.GlobalUpdate


class GlobalDeltaAggregationFactory(IGlobalMethodFactory):
    """Method for returning the Delta aggregation method"""

    def get_global_method(self):
        return delta_aggregation.GlobalUpdate


class GlobalDeltaAggregationFedDynFactory(IGlobalMethodFactory):
    """Method for returning the FedDyn global aggregation method"""

    def get_global_method(self):
        return delta_aggregation_fedDyn.GlobalUpdate


class GlobalDeltaAggregationFedAGMFactory(IGlobalMethodFactory):
    """Method for returning the FedAGM global aggregation method"""

    def get_global_method(self, args, device, trainset, testloader, valloader, local_update):
        return delta_aggregation_AGM.FedAGMGlobalUpdate(args=args, device=device, trainset=trainset,
                                                        testloader=testloader, valloader=valloader,
                                                        local_update=local_update)


class GlobalBaseAggregationFedSAFactory(IGlobalMethodFactory):
    """Method for returning the FedSA global aggregation method"""

    def get_global_method(self):
        return base_aggregation_fedSA.GlobalUpdate


class GlobalDeltaAggregationFedSAFactory(IGlobalMethodFactory):
    """Method for returning the Delta FedSA global aggregation method"""

    def get_global_method(self):
        return delta_aggregation_AGM_fedSA.GlobalUpdate


class GlobalBaseAggregationFedSAIGFactory(IGlobalMethodFactory):
    """Method for returning the Delta FedSA global aggregation method"""

    def get_global_method(self):
        return base_aggregation_fedSA_IG.GlobalUpdate


class GlobalDeltaAggregationFedSAIGFactory(IGlobalMethodFactory):
    """Method for returning the Delta FedSA global aggregation method"""

    def get_global_method(self):
        return delta_aggregation_AGM_fedSA_IG.GlobalUpdate


class GlobalBaseAggregationReputationFactory(IGlobalMethodFactory):
    """Method for returning the Delta FedSA global aggregation method"""

    def get_global_method(self, args, device, trainset, testloader, valloader, local_update):
        return base_aggregation_reputation.ReputationGlobalUpdate(args=args, device=device, trainset=trainset,
                                                        testloader=testloader, valloader=valloader,
                                                        local_update=local_update)


class GlobalBaseAggregationOortFactory(IGlobalMethodFactory):
    """Method for returning the Base global aggregation method with Oort baseline"""

    def get_global_method(self, args, device, trainset, testloader, valloader, local_update):
        return base_aggregation_Oort.OortGlobalUpdate(args=args, device=device, trainset=trainset,
                                                        testloader=testloader, valloader=valloader,
                                                        local_update=local_update)


class GlobalDeltaAggregationFedAGMIGFactory(IGlobalMethodFactory):
    """Method for returning the Delta global with IG aggregation method"""

    def get_global_method(self, args, device, trainset, testloader, valloader, local_update):
        return delta_aggregation_AGM_IG.DeltaFedSBSGlobalUpdate(args=args, device=device, trainset=trainset,
                                                        testloader=testloader, valloader=valloader,
                                                        local_update=local_update)


class LocalBaseFactory(ILocalMethodFactory):
    """Method for returning the FedAVG local update method"""

    def get_local_method(self):
        return base.LocalUpdate


class LocalFedProxFactory(ILocalMethodFactory):
    """Method for returning the FedAVG with L2 local update method"""

    def get_local_method(self):
        return weight_l2.LocalUpdate


class LocalFedCMFactory(ILocalMethodFactory):
    """Method for returning the FedCM local update method"""

    def get_local_method(self):
        return fedCM.LocalUpdate


class LocalFedDynFactory(ILocalMethodFactory):
    """Method for returning the FedDyn local update method"""

    def get_local_method(self):
        return fedDyn.LocalUpdate


class LocalFedAGMFactory(ILocalMethodFactory):
    """Method for returning the FedAGM local update method"""

    def get_local_method(self):
        return fedAGM.LocalUpdate


LOCALUPDATE_LOOKUP_TABLE = {'fedavg': LocalBaseFactory(),
                            'fedprox': LocalFedProxFactory(),
                            'fedcm': LocalFedCMFactory(),
                            'feddyn': LocalFedDynFactory(),
                            'fedagm':  LocalFedAGMFactory()
                            }


GLOBALAGGREGATION_LOOKUP_TABLE = {'fedavg': GlobalBaseAggregationFactory(),
                                  'slowmo': GlobalDeltaAggregationSlowmoFactory(),
                                  'global_adam': GlobalAdamAggregationFactory(),
                                  'global_delta': GlobalDeltaAggregationFactory(),
                                  'feddyn': GlobalDeltaAggregationFedDynFactory(),
                                  'fedagm':  GlobalDeltaAggregationFedAGMFactory(),
                                  'fedagm_ig':  GlobalDeltaAggregationFedAGMIGFactory(),
                                  'fedsa':  GlobalBaseAggregationFedSAFactory(),
                                  'fedsa_agm': GlobalDeltaAggregationFedSAFactory(),
                                  "fedavg_ig": GlobalBaseAggregationIGFactory(),
                                  "fedsa_ig": GlobalBaseAggregationFedSAIGFactory(),
                                  "fedsa_agm_ig": GlobalDeltaAggregationFedSAIGFactory(),
                                  "fedavg_reputation": GlobalBaseAggregationReputationFactory(),
                                  "fedavg_oort": GlobalBaseAggregationOortFactory()
                                  }

