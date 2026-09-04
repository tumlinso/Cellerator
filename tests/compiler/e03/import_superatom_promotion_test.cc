#include <Cellerator/compiler/program/import_superatom_promotion_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){superatom_candidate_v1 c{"ab","pbmc","compose(a,b)","split(ab)->a,b","bench42",8,10,true,true};assert(evaluate_superatom_promotion_v1(c,"pbmc").promoted);auto slow=c;slow.total_cost=11;auto r=evaluate_superatom_promotion_v1(slow,"pbmc");assert(!r.promoted&&r.reason=="complete cost does not win");auto wrong=c;assert(!evaluate_superatom_promotion_v1(wrong,"other").promoted);auto lossy=c;lossy.deconstruction.clear();assert(!evaluate_superatom_promotion_v1(lossy,"pbmc").promoted);}
