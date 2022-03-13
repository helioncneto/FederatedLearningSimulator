from abc import ABC, abstractmethod
import global_update_method.base_aggregation as base_aggregation
import global_update_method.delta_aggregation_slowmo as delta_aggregation_slowmo
import global_update_method.adam_aggregation as adam_aggregation
import global_update_method.delta_aggregation as delta_aggregation
import global_update_method.delta_aggregation_fedDyn as delta_aggregation_fedDyn
import global_update_method.delta_aggregation_AGM as delta_aggregation_AGM

import local_update_method.base as base
import local_update_method.weight_l2 as weight_l2
import local_update_method.fedCM as fedCM
import local_update_method.fedDyn as fedDyn
import local_update_method.fedAGM as fedAGM


class IGlobalMethodFactory(ABC):
    """Abstract class for global method classes"""

    @abstractmethod
    def get_global_method(self):
        """Return the global method class"""


class ILocalMethodFactory(ABC):
    """Abstract class for global method classes"""

    @abstractmethod
    def get_local_method(self):
        """Return the global method class"""


class GlobalBaseAggregationFactory(IGlobalMethodFactory):
    """Method for returning the FedAVG global aggregation method"""

    def get_global_method(self):
        return base_aggregation.GlobalUpdate


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

    def get_global_method(self):
        return delta_aggregation_AGM.GlobalUpdate


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


LOCALUPDATE_LOOKUP_TABLE = {'Fedavg': LocalBaseFactory(),
                            'FedProx': LocalFedProxFactory(),
                            'FedCM': LocalFedCMFactory(),
                            'FedDyn': LocalFedDynFactory(),
                            'FedAGM':  LocalFedAGMFactory()
                            }


GLOBALAGGREGATION_LOOKUP_TABLE = {'base_avg': GlobalBaseAggregationFactory(),
                                  'SlowMo': GlobalDeltaAggregationSlowmoFactory(),
                                  'global_adam': GlobalAdamAggregationFactory(),
                                  'global_delta': GlobalDeltaAggregationFactory(),
                                  'FedDyn': GlobalDeltaAggregationFedDynFactory(),
                                  'FedAGM':  GlobalDeltaAggregationFedAGMFactory()
                                  }

if __name__ == "__main__":
    print(GLOBALAGGREGATION_LOOKUP_TABLE['base_avg'].get_global_method())
