#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
struct template_instantiation_v1{std::string template_name,numeric_type,domain,profile,backend,symbol;std::uint64_t body_hash=0;};
struct template_deduplication_v1{std::vector<std::uint32_t>canonical_for_input;std::vector<template_instantiation_v1>canonical;};
enum class template_dedup_status_v1:std::uint8_t{valid=0,invalid_instantiation,odr_conflict};
[[nodiscard]] template_dedup_status_v1 deduplicate_template_instantiations_v1(const std::vector<template_instantiation_v1>&,template_deduplication_v1*)noexcept;
}
