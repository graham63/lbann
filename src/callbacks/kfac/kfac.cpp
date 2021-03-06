////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <sstream>

#include "lbann/utils/profiling.hpp"
#include "lbann/callbacks/kfac/kfac.hpp"
#include "lbann/callbacks/kfac/kfac_util.hpp"
#include "lbann/callbacks/kfac/kfac_block_fc_conv.hpp"
#include "lbann/callbacks/kfac/kfac_block_bn.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"

namespace lbann {
namespace callback {

template <El::Device Device>
void kfac<Device>::setup(model *m) {
  m_rank = m->get_comm()->get_rank_in_trainer();

  const auto v2s =
      [](const std::vector<double> v) {
        std::ostringstream oss;
        for(auto i = v.begin(); i != v.end(); i++) {
          if(i != v.begin())
            oss << ",";
          oss << *i;
        }
        return oss.str();
      };

  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    std::ostringstream oss;
    oss << "K-FAC callback setup:"
        << " damping_act=" << v2s(m_damping_act_params)
        << " damping_err=" << v2s(m_damping_err_params)
        << " damping_bn_act=" << v2s(m_damping_bn_act_params)
        << " damping_bn_err=" << v2s(m_damping_bn_err_params)
        << " damping_warmup_steps=" << m_damping_warmup_steps
        << " kronecker_decay=" << m_kronecker_decay
        << " learning_rate_factor=" << m_learning_rate_factor
        << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac<Device>::on_epoch_end(model *m) {
  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
    const auto epoch = c.get_epoch();
    std::ostringstream oss;
    oss << "K-FAC callback: damping_value="
        << m_damping_act << " (act)"
        << ", " << m_damping_err << " (err)"
        << ", " << m_damping_bn_act << " (bn_act)"
        << ", " << m_damping_bn_err << " (bn_err)"
        << ", update_interval=" << m_update_interval
        << " at " << epoch << " epochs"
        << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac<Device>::on_backward_prop_end(model *m) {
  // Update the damping value
  // using a modified Tikhonov damping tequnique from
  // http://arxiv.org/abs/1811.12019
  const auto get_next_damping =
      [](const double damping_prev,
         const std::vector<double> damping_params,
         const double damping_warmup_steps) {
        if(damping_params.size() == 1)
          return damping_params[0];
        const DataType alpha = 2.0 * log10(damping_params[0] / damping_params[1]) / damping_warmup_steps;
        return (1.0-alpha) * damping_prev + alpha * damping_params[1];
      };
  m_damping_act = get_next_damping(
      m_damping_act, m_damping_act_params, m_damping_warmup_steps);
  m_damping_err = get_next_damping(
      m_damping_err, m_damping_err_params, m_damping_warmup_steps);
  m_damping_bn_act = get_next_damping(
      m_damping_bn_act, m_damping_bn_act_params, m_damping_warmup_steps);
  m_damping_bn_err = get_next_damping(
      m_damping_bn_err, m_damping_bn_err_params, m_damping_warmup_steps);

  // Update the udpate interval
  if(m_update_intervals.size() == 1)
    m_update_interval = m_update_intervals[0];
  else {
    const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
    const auto num_steps = c.get_step();
    m_update_interval = m_update_intervals[0]
        + ((double) m_update_intervals[1]-m_update_intervals[0])
        * std::min((double) num_steps/ m_update_interval_steps, 1.0);
  }

  // Get some configs
  const auto comm = m->get_comm();
  const auto& context = static_cast<const sgd_execution_context&>(m->get_execution_context());
  const size_t num_steps = context.get_step();
  const auto layers = m->get_layers();

  // List up layers to be updated
  if(m_blocks.size() == 0){
    prof_region_begin("kfac-setup", prof_color, prof_sync);
    const size_t num_procs = comm->get_procs_per_trainer();
    std::unordered_map<std::string, int> proc_ranks;
    for(auto i_layer = layers.begin(); i_layer != layers.end(); i_layer++) {
      const size_t layer_id = std::distance(layers.begin(), i_layer);
      const auto &l = *i_layer;
      const auto l_fc = dynamic_cast<fully_connected_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const auto l_conv = dynamic_cast<convolution_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const auto l_bn = dynamic_cast<batch_normalization_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const bool is_fc = (l_fc != nullptr);
      const bool is_conv = (l_conv != nullptr);
      const bool is_bn = (l_bn != nullptr);
      if(!(is_fc || is_conv || is_bn))
        continue;

      if(std::find(m_disable_layers.begin(), m_disable_layers.end(), l->get_name()) != m_disable_layers.end()) {
        if(comm->am_trainer_master())
          std::cout << "K-fac callback: " << l->get_name() << " is ignored to optimize with K-FAC." << std::endl;
        continue;
      }

      prof_region_begin(("kfac-setup/" + l->get_name()).c_str(), prof_color, prof_sync);

      // Ignore layers without optimizers
      const auto& weights = l->get_weights(0);
      const optimizer *w_optimizer = weights.get_optimizer();
      if(w_optimizer == nullptr)
        continue;

      std::string proc_rank_key = "all";
      if(m_inverse_strategy == kfac_inverse_strategy::EACH)
        proc_rank_key = (is_fc ? "fc" : (is_conv ? "conv" : "bn"));
      if(proc_ranks.find(proc_rank_key) == proc_ranks.end())
        proc_ranks[proc_rank_key] = 0;
      int& proc_rank = proc_ranks[proc_rank_key];

      // Check layer property
      if(l->get_num_parents() != 1 || l->get_num_children() != 1) {
        std::stringstream err;
        err << "The K-FAC callback only supports layers who have exact one parent and child."
            << " layer: " << l->get_name()
            << ", #parent: " << l->get_num_parents()
            << ", #child: " << l->get_num_children();
        LBANN_ERROR(err.str());
      }

      std::shared_ptr<kfac_block<Device>> block;
      if(is_fc || is_conv) {
        block = std::make_shared<kfac_block_fc_conv<Device>>(
            l, this, layer_id, proc_rank, is_conv);
      } else if(is_bn) {
        block = std::make_shared<kfac_block_bn<Device>>(
            l, this, layer_id, proc_rank);
      }

      m_blocks.push_back(std::move(block));
      if(m_inverse_strategy != kfac_inverse_strategy::ROOT)
        proc_rank = (proc_rank+1)%num_procs;

      prof_region_end(("kfac-setup/" + l->get_name()).c_str(), prof_sync);
    }

    if(comm->am_trainer_master()) {
      for(const auto& block : m_blocks)
        std::cout << "K-FAC callback setup: "
                  << block->get_info() << std::endl;
    }

    prof_region_end("kfac-setup", prof_sync);
  }

  prof_region_begin("kfac-step", prof_color, prof_sync);

  // Step 1: Ensure that each process has averaged Kronecker factors
  // for the model-parallel part.
  const bool is_first_step = (!m_has_kronecker_inverse);
  const bool is_kronecker_update_required =
      ((num_steps%m_update_interval) == 0 || !m_has_kronecker_inverse);
  if(is_kronecker_update_required) {
    prof_region_begin("kfac-update", prof_color, prof_sync);

    prof_region_begin("kfac-update/local", prof_color, prof_sync);
    for(auto& block : m_blocks) {
      prof_region_begin(("kfac-update/local/" + block->get_name()).c_str(), prof_color, prof_sync);
      block->compute_local_kronecker_factors(
          comm, m_print_matrix, m_print_matrix_summary);
      prof_region_end(("kfac-update/local/" + block->get_name()).c_str(), prof_sync);
    }
    prof_region_end("kfac-update/local", prof_sync);

#ifdef LBANN_NVPROF
    prof_region_begin("kfac-update/local-barrier", prof_color, prof_sync);
    CHECK_CUDA(cudaDeviceSynchronize());
    comm->trainer_barrier();
    prof_region_end("kfac-update/local-barrier", prof_sync);
#endif // LBANN_NVPROF

    // List-up buffers to synchronize.
    std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> buffers;
    size_t global_buffer_size = 0;
    for(auto& block : m_blocks)
      for(auto L : block->get_local_kronecker_buffers()) {
        const size_t rank = block->get_inverse_proc_rank();
        buffers.emplace_back(rank, L);
        assert(L->Width() == 1);
        global_buffer_size += L->Height();
      }

    // Perform reduce-scatter.
    prof_region_begin("kfac-update/reduce-scatter", prof_color, prof_sync);
    const auto reduce_scatter_mode = kfac_reduce_scatter_mode::ALLREDUCE;
    El::Matrix<DataType, Device>& global_buffer =
        get_workspace_matrix(
            "reduce_scatter_send_buffer",
            kfac_util::is_reduce_scatter_buffer_required(reduce_scatter_mode) ? global_buffer_size : 0,
            1);
    kfac_util::reduce_scatter_blocks(
        buffers, global_buffer, comm, reduce_scatter_mode);
    prof_region_end("kfac-update/reduce-scatter", prof_sync);

#ifdef LBANN_NVPROF
    prof_region_begin("kfac-update/reduce-scatter-barrier", prof_color, prof_sync);
    CHECK_CUDA(cudaDeviceSynchronize());
    comm->trainer_barrier();
    prof_region_end("kfac-update/reduce-scatter-barrier", prof_sync);
#endif // LBANN_NVPROF

    prof_region_begin("kfac-update/average", prof_color, prof_sync);
    for(auto& block : m_blocks) {
      prof_region_begin(("kfac-update/average/" + block->get_name()).c_str(), prof_color, prof_sync);
      block->update_kronecker_average(
          comm,
          m_kronecker_decay,
          m_print_matrix, m_print_matrix_summary);
      prof_region_end(("kfac-update/average/" + block->get_name()).c_str(), prof_sync);
    }
    prof_region_end("kfac-update/average", prof_sync);

    prof_region_end("kfac-update", prof_sync);
  }

  // Step 2: Model-parallel inverse computation
  prof_region_begin("kfac-inverse", prof_color, prof_sync);
  for(auto& block : m_blocks) {
    if(!is_kronecker_update_required || (size_t) comm->get_rank_in_trainer() != block->get_inverse_proc_rank())
      continue;

    prof_region_begin(("kfac-inverse/" + block->get_name()).c_str(), prof_color, prof_sync);
    // TODO: Add kfac_block::is_bn?
    const bool is_bn = dynamic_cast<kfac_block_bn<Device>*>(block.get()) != nullptr;
    block->update_kronecker_inverse(
        comm, m_use_pi,
        is_bn ? m_damping_bn_act : m_damping_act,
        is_bn ? m_damping_bn_err : m_damping_err,
        m_learning_rate_factor,
        m_print_matrix, m_print_matrix_summary,
        m_print_time);
    prof_region_end(("kfac-inverse/" + block->get_name()).c_str(), prof_sync);
  }
  m_has_kronecker_inverse = true;
  prof_region_end("kfac-inverse", prof_sync);

#ifdef LBANN_NVPROF
  prof_region_begin("kfac-inverse-barrier", prof_color, prof_sync);
  CHECK_CUDA(cudaDeviceSynchronize());
  comm->trainer_barrier();
  prof_region_end("kfac-inverse-barrier", prof_sync);
#endif // LBANN_NVPROF

  // Step 3: All-gather of each preconditioned gradient tensor
  prof_region_begin("kfac-allgather", prof_color, prof_sync);
  {
    // List-up buffers to synchronize.
    std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> buffers;
    int local_buffer_size = 0, global_buffer_size = 0;
    for(auto& block : m_blocks)
      for(auto L : block->get_preconditioned_grad_buffers()) {
        const size_t rank = block->get_inverse_proc_rank();
        buffers.emplace_back(rank, L);
        assert(L->Width() == 1);
        if(rank == (size_t) comm->get_rank_in_trainer())
          local_buffer_size += L->Height();
        global_buffer_size += L->Height();
      }

    // Perform allgather.
    const auto allgather_mode = kfac_allgather_mode::ALLREDUCE;
    const auto is_buffer_needed = kfac_util::is_allgather_buffer_required(allgather_mode);
    El::Matrix<DataType, Device>& local_buffer =
        get_workspace_matrix(
            "allgather_send_buffer",
            is_buffer_needed.first ? local_buffer_size : 0,
            1);
    El::Matrix<DataType, Device>& global_buffer =
        get_workspace_matrix(
            "allgather_recv_buffer",
            is_buffer_needed.second ? global_buffer_size : 0,
            1);
    kfac_util::allgather_blocks(
        buffers, local_buffer, global_buffer, comm, allgather_mode);
  }
  prof_region_end("kfac-allgather", prof_sync);

#ifdef LBANN_NVPROF
  prof_region_begin("kfac-allgather-barrier", prof_color, prof_sync);
  CHECK_CUDA(cudaDeviceSynchronize());
  comm->trainer_barrier();
  prof_region_end("kfac-allgather-barrier", prof_sync);
#endif // LBANN_NVPROF

  prof_region_end("kfac-step", prof_sync);

  if(is_first_step) {
    for(auto& block : m_blocks) {
      for(auto& info : block->get_internal_matrix_info()) {
        std::ostringstream oss;
        oss << "K-FAC callback matrix allocation (rank="
            << comm->get_rank_in_trainer()
            << "): " << block->get_name()
            << " " << std::get<0>(info)
            << " (" << std::get<1>(info)
            << "x" << std::get<2>(info)
            << ")" << std::endl;
        std::cout << oss.str();
      }
    }
  }
}

template <El::Device Device>
El::Matrix<DataType, Device>& kfac<Device>::get_workspace_matrix(
    const std::string& key, const size_t height, const size_t width) {
  if(m_workspace.find(key) == m_workspace.end()) {
    std::ostringstream oss;
    oss << "K-FAC callback workspace allocation (rank=" << m_rank
        << "): " << key << " (" << height << "x" << width << ")" << std::endl;
    std::cout << oss.str();
    m_workspace.emplace(
        key, El::Matrix<DataType, Device>(height, width));
#ifdef HYDROGEN_HAVE_CUB
    m_workspace[key].SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
  }
  auto& ret = m_workspace[key];
  if((size_t) ret.Height() != height || (size_t) ret.Width() != width) {
    // Make sure that no kernels are using this workspace.
    El::Synchronize(El::SyncInfoFromMatrix(ret));
    ret.Resize(height, width);
  }
  return ret;
}

std::unique_ptr<callback_base>
build_kfac_callback_from_pbuf(
    const google::protobuf::Message& proto_msg,
    const std::shared_ptr<lbann_summary>&) {
  using MsgType = lbann_data::Callback::CallbackKFAC;
#ifdef LBANN_HAS_GPU
  using CallbackType = kfac<El::Device::GPU>;
#else
  using CallbackType = kfac<El::Device::CPU>;
#endif // LBANN_HAS_GPU
  const auto& params = dynamic_cast<const MsgType&>(proto_msg);

  const auto parse_damping_params =
      [](const std::string str) {
        if(str == "")
          return std::vector<double>({CallbackType::damping_0_default});
        else {
          const auto ret = parse_list<double>(str);
          if(ret.size() > 2)
            LBANN_ERROR("The length of damping vectors should be 1 or 2.");
          return ret;
        }
      };

  const auto parse_update_intervals =
      [](const std::string str) {
        if(str == "")
          return std::vector<size_t>({1});
        else {
          const auto ret = parse_list<size_t>(str);
          if(ret.size() > 2)
            LBANN_ERROR("The length of update interval vectors should be 1 or 2.");
          return ret;
        }
      };

  const std::vector<double> damping_act_params = parse_damping_params(params.damping_act());
  const std::vector<double> damping_err_params = parse_damping_params(params.damping_err());
  const std::vector<double> damping_bn_act_params = parse_damping_params(params.damping_bn_act());
  const std::vector<double> damping_bn_err_params = parse_damping_params(params.damping_bn_err());
  size_t damping_warmup_steps = params.damping_warmup_steps();
  if(damping_warmup_steps == 0) damping_warmup_steps = CallbackType::damping_warmup_steps_default;
  double kronecker_decay = params.kronecker_decay();
  if(kronecker_decay == 0.0)
    kronecker_decay = CallbackType::kronecker_decay_default;
  const bool print_time = params.print_time();
  const bool print_matrix = params.print_matrix();
  const bool print_matrix_summary = params.print_matrix_summary();
  const bool use_pi = params.use_pi();
  const std::vector<size_t> update_intervals = parse_update_intervals(params.update_intervals());
  const size_t update_interval_steps = params.update_interval_steps();

  const std::string inverse_strategy_str = params.inverse_strategy();
  kfac_inverse_strategy inverse_strategy;
  if(inverse_strategy_str == "" || inverse_strategy_str == "all")
    inverse_strategy = kfac_inverse_strategy::ALL;
  else if(inverse_strategy_str == "each")
    inverse_strategy = kfac_inverse_strategy::EACH;
  else if(inverse_strategy_str == "root")
    inverse_strategy = kfac_inverse_strategy::ROOT;
  else {
    std::stringstream err;
    err << "Invalid inverse strategy type: "
        << inverse_strategy_str;
    LBANN_ERROR(err.str());
  }

  const std::vector<std::string> disable_layers =
      parse_list<std::string>(params.disable_layers());

  double learning_rate_factor = params.learning_rate_factor();
  if(learning_rate_factor == 0.0)
    learning_rate_factor = 1.0;

  return make_unique<CallbackType>(
      damping_act_params,
      damping_err_params,
      damping_bn_act_params,
      damping_bn_err_params,
      damping_warmup_steps,
      kronecker_decay,
      print_time, print_matrix, print_matrix_summary,
      use_pi,
      update_intervals, update_interval_steps,
      inverse_strategy,
      disable_layers,
      learning_rate_factor);
}

template class kfac<El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class kfac<El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace callback
} // namespace lbann
