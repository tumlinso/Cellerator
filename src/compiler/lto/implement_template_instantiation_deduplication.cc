#include <Cellerator/compiler/lto/implement_template_instantiation_deduplication_v1.hh>
#include <map>
#include <tuple>
namespace cellerator::compiler::lto::v1 {
template_dedup_status_v1 deduplicate_template_instantiations_v1(const std::vector<template_instantiation_v1>&xs,template_deduplication_v1*r)noexcept{if(!r)return template_dedup_status_v1::invalid_instantiation;r->canonical.clear();r->canonical_for_input.clear();std::map<std::tuple<std::string,std::string,std::string,std::string,std::string>,std::uint32_t>seen;for(const auto&x:xs){if(x.template_name.empty()||x.numeric_type.empty()||x.domain.empty()||x.profile.empty()||x.backend.empty()||x.symbol.empty()||!x.body_hash)return template_dedup_status_v1::invalid_instantiation;auto k=std::make_tuple(x.template_name,x.numeric_type,x.domain,x.profile,x.backend);auto [it,add]=seen.emplace(k,r->canonical.size());if(add)r->canonical.push_back(x);else if(r->canonical[it->second].body_hash!=x.body_hash)return template_dedup_status_v1::odr_conflict;r->canonical_for_input.push_back(it->second);}return template_dedup_status_v1::valid;}
}
