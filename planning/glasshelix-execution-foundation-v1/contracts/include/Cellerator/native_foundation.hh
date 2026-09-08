#pragma once
#ifndef CE_NF1_DECLARATION_CHECK_ONLY
#error "Planning declarations only. Implement real Cellerator facilities; do not install this file."
#endif
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
namespace cellerator::native_foundation {
enum class device {host,cuda};
enum class op {argument,constant,add,multiply,divide,tanh};
struct term {op code;std::uint32_t left=0,right=0;double scalar=0;};
struct nary_definition {std::span<const std::uint32_t> argument_indices;std::span<const term> expression;std::size_t arity;};
struct counts {std::size_t structure_preparations,value_instances,hot_allocations;};
struct event {void wait()const;};
struct result {event ready;std::vector<float> download()const;};
struct value_instance {std::uint64_t id;};
struct session {struct impl;std::unique_ptr<impl> p;explicit session(device);~session();device actual_device()const;};
struct prepared_nary {
 struct impl;std::unique_ptr<impl> p;
 prepared_nary(session&,const nary_definition&,std::size_t width);~prepared_nary();
 value_instance bind_gain(float);result evaluate(value_instance,std::span<const float> arguments);
 result gain_jvp(value_instance,std::span<const float> arguments,float direction);
 counts inspect()const;
};
}
