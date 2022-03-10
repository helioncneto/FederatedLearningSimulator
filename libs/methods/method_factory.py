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

    def get_global_method(self):
        return base_aggregation.GlobalUpdate


class GlobalDeltaAggregationSlowmoFactory(IGlobalMethodFactory):

    def get_global_method(self):
        return delta_aggregation_slowmo.GlobalUpdate


class GlobalAdamAggregationFactory(IGlobalMethodFactory):

    def get_global_method(self):
        return adam_aggregation.GlobalUpdate


class GlobalDeltaAggregationFactory(IGlobalMethodFactory):

    def get_global_method(self):
        return delta_aggregation.GlobalUpdate


class GlobalDeltaAggregationFedDynFactory(IGlobalMethodFactory):

    def get_global_method(self):
        return delta_aggregation_fedDyn.GlobalUpdate


class GlobalDeltaAggregationFedAGM(IGlobalMethodFactory):

    def get_global_method(self):
        return delta_aggregation_AGM.GlobalUpdate


class LocalBase(ILocalMethodFactory):

    def get_local_method(self):
        return base.LocalUpdate


