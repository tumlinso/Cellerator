#include <Cellerator/compiler/tooling/implement_document_scheduling_and_cancellation_v1.hh>

int main() {
    Cellerator::compiler::tooling::document_scheduler_v1 scheduler;
    scheduler.edited("stdin.cell", 1, 0, true);
    return scheduler.next_basic(50) ? 0 : 1;
}
