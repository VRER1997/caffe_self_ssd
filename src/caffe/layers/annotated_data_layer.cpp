#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
namespace caffe {

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param),
    reader_ (param) {}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer(){
  this->StopInternalThread();
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int batch_size = this->layer_param.data_param().batch_size();
    const AnnotatedDataParameter& anno_data_param = 
        this->layer_param.annotated_data_param();
    for(int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
        batch_samplers_.push_back(annot_data_param.batch_sampler(i));
    }
    label_map_file = anno_data_param.label_map_file();
    const TransformationParameter& transform_param = 
        this->layer_param_.transform_param();
    
  
}

}
