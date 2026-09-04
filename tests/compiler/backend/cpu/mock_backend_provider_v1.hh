#pragma once

#include <Cellerator/compiler/backend/freeze_the_backend_provider_abi_v1.hh>

[[nodiscard]] cellerator::compiler::backend::v1::backend_provider_v1
make_mock_backend_provider_v1() noexcept;
