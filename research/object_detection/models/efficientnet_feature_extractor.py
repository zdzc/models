import tensorflow as tf

import efficientnet.tfkeras as efn

from object_detection.utils import ops, shape_utils
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators


EFFICIENTNET_CLASS_MAPPING = {
    'efficientnet-b0': efn.EfficientNetB0,
}

EFFICIENTNET_NODES_MAPPING = {
    'efficientnet-b0': ["block3b_add", "block5c_add", "block7a_project_bn", "", "", "", ""],
}


class SSDEfficientNetFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor):
    """ SSD Feature Extractor using EfficientNet """

    def __init__(self,
                 is_training,
                 depth_multiplier,
                 min_depth,
                 pad_to_multiple,
                 conv_hyperparams,
                 freeze_batchnorm,
                 inplace_batchnorm_update,
                 network_version='efficientnet-b0',
                 min_feature_level=3,
                 max_feature_level=7,
                 additional_layer_depth=None,
                 reuse_weights=None,
                 use_explicit_padding=None,
                 use_depthwise=False,
                 use_antialias=False,
                 override_base_feature_extractor_hyperparams=False,
                 name=None,
                 data_format="channels_last"):
        """EfficientNet Feature Extractor for SSD Models

        Arguments:
            is_training {bool} -- wether the network is in training mode
            depth_multiplier {float} -- depth multiplier for feature extractor 
            min_depth {int32} -- minimum feature extractor depth
            pad_to_multiple {int32} -- the nearest multiple to zero pad the input height and 
                width dimensions to 
            conv_hyperparams -- A `hyperparams_builder.KerasLayerHyperparams` object 
                containing convolution hyperparameters for the layers added on top of
                the base feature extractor
            inplace_batchnorm_update --  Whether to update batch norm moving average
               values inplace. When this is false train op must add a control
               dependency on tf.graphkeys.UPDATE_OPS collection in order to update
               batch norm statistics.

        Keyword Arguments:
            reuse_weights -- whether to reuse variables. Default is None 
            use_explicit_padding -- whether to use depthwise convolutions. Default is False
            use_depthwise -- whether to use depthwise convolution
            override_base_feature_extractor_hyperparams -- Whether to override
               hyperparameters of the base feature extractor with the one from
               `conv_hyperparams`.
            name --  A string name scope to assign to the model. If 'None', Keras
               will auto-generate one from the class name
            data_format -- channels_last or channels_first
        """

        if network_version not in EFFICIENTNET_CLASS_MAPPING:
            raise ValueError('Network version not supported')

        super(SSDEfficientNetFeatureExtractor, self).__init__(
            is_training=is_training,
            depth_multiplier=depth_multiplier,
            min_depth=min_depth,
            pad_to_multiple=pad_to_multiple,
            conv_hyperparams=conv_hyperparams,
            freeze_batchnorm=freeze_batchnorm,
            inplace_batchnorm_update=inplace_batchnorm_update,
            use_explicit_padding=use_explicit_padding,
            use_depthwise=use_depthwise,
            override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
            name=name
        )
        self._data_format = data_format
        self._min_feature_level = min_feature_level
        self._max_feature_level = max_feature_level
        self._additional_layer_depth = additional_layer_depth
        self._network_name = network_version

        default_nodes = EFFICIENTNET_NODES_MAPPING[network_version]
        default_nodes_depth = [-1, -1, -1, 512, 256, 256, 128]
        self._used_nodes = default_nodes[min_feature_level -
                                         3:max_feature_level-2]
        self._used_nodes_depth = default_nodes_depth[min_feature_level -
                                                     3:max_feature_level-2]
        self._feature_map_layout = {
            'from_layer': self._used_nodes,
            'layer_depth': self._used_nodes_depth,
            'use_depthwise': self._use_depthwise,
            'use_explicit_padding': self._use_explicit_padding
        }
        self._feature_map_generator = None
        self.net = None

    def build(self, input_shape):
        efficientnet = EFFICIENTNET_CLASS_MAPPING[self._network_name]
        model = efficientnet(include_top=False, weights=None)

        outputs = [model.get_layer(x).output for x in self._used_nodes if x]
        self.net = tf.keras.Model(inputs=model.input, outputs=outputs)

        # feature map generator
        self._feature_map_generator = feature_map_generators.KerasMultiResolutionFeatureMaps(
            feature_map_layout=self._feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            is_training=self._is_training,
            conv_hyperparams=self._conv_hyperparams,
            freeze_batchnorm=self._freeze_batchnorm,
            name=None
        )
        self.built = True

    def preprocess(self, resized_inputs):
        """SSD preprocessing: map pixel values ot range [-1, 1]"""
        return resized_inputs * (2.0 / 255.0) - 1.0

    def _extract_features(self, preprocessed_inputs):
        """Extract features from preprocessed inputs"""
        preprocessed_inputs = shape_utils.check_min_image_dim(
            33, preprocessed_inputs)

        image_features = self.net(ops.pad_to_multiple(
            preprocessed_inputs, self._pad_to_multiple))
        layouts = {self._used_nodes[i]: image_features[i]
                   for i, x in enumerate(self._used_nodes) if x}
        feature_maps = self._feature_map_generator(layouts)
        if self._additional_layer_depth:
            final_feature_map = []
            for idx, feature in enumerate(feature_maps.values()):
                feature = l.Conv2D(filters=self._additional_layer_depth,
                                   kernel_size=1,
                                   strides=[1, 1],
                                   use_bias=True,
                                   data_format=self._data_format,
                                   name='conv1x1_'+str(idx))(feature)
                feature = l.BatchNormalization()(feature, training=self._is_training)
                feature = l.ReLU(max_value=6)(feature)
                final_feature_map.append(feature)
            return final_feature_map
        else:
            return list(feature_maps.values())
