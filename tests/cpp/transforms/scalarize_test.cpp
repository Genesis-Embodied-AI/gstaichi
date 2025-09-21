#include "gtest/gtest.h"

#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace gstaichi::lang {

TEST(Scalarize, ScalarizeGlobalStore) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = ExternalPtrStmt()
    TensorType<4 x i32>  %2 = MatrixInitStmt([1, 1, 2, 2])
    StoreStmt(%1, %2)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  auto const_1_stmt = block->push_back<ConstStmt>(TypedConstant(1));
  auto const_2_stmt = block->push_back<ConstStmt>(TypedConstant(2));
  auto type =
      TypeFactory::get_instance().get_ndarray_struct_type(tensor_type, 1);

  auto argload_stmt = block->push_back<ArgLoadStmt>(
      std::vector<int>{0} /*arg_id*/, type, /*is_ptr*/ true,
      /*create_load*/ false);

  std::vector<Stmt *> indices = {};
  Stmt *dest_stmt = block->push_back<ExternalPtrStmt>(
      argload_stmt, indices);  // fake ExternalPtrStmt

  dest_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  std::vector<Stmt *> matrix_init_vals = {const_1_stmt, const_1_stmt,
                                          const_2_stmt, const_2_stmt};
  auto matrix_init_stmt =
      block->push_back<MatrixInitStmt>(std::move(matrix_init_vals));
  matrix_init_stmt->ret_type = tensor_type;

  block->push_back<GlobalStoreStmt>(dest_stmt, matrix_init_stmt);

  irpass::scalarize(block.get());
  irpass::lower_matrix_ptr(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 2 /*const*/ + 1 /*argload*/ + 4 /*const*/ +
                               4 /*external_ptr*/ + 4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[3]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[4]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<GlobalStoreStmt>(), true);

  EXPECT_EQ(block->statements[6]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[8]->is<GlobalStoreStmt>(), true);

  EXPECT_EQ(block->statements[9]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[10]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[11]->is<GlobalStoreStmt>(), true);

  EXPECT_EQ(block->statements[12]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[13]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[14]->is<GlobalStoreStmt>(), true);
}

TEST(Scalarize, ScalarizeGlobalLoad) {
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = ExternalPtrStmt()
    TensorType<4 x i32>  %2 = LoadStmt(%1)
    StoreStmt(%1, %2)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  auto type =
      TypeFactory::get_instance().get_ndarray_struct_type(tensor_type, 1);

  auto argload_stmt = block->push_back<ArgLoadStmt>(
      std::vector<int>{0} /*arg_id*/, type, /*is_ptr*/ true,
      /*create_load*/ false);

  std::vector<Stmt *> indices = {};
  Stmt *src_stmt = block->push_back<ExternalPtrStmt>(
      argload_stmt, indices);  // fake ExternalPtrStmt
  src_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  auto load_stmt = block->push_back<GlobalLoadStmt>(src_stmt);

  // Without this GlobalStoreStmt, nothing survives irpass::die()
  block->push_back<GlobalStoreStmt>(src_stmt, load_stmt);

  irpass::scalarize(block.get());
  irpass::lower_matrix_ptr(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 1 /*argload*/ + 4 /*const*/ + 4 /*external_ptr*/ +
                               4 /*load*/ + 4 /*const*/ + 4 /*external_ptr*/ +
                               4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[1]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<GlobalLoadStmt>(), true);

  EXPECT_EQ(block->statements[4]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[6]->is<GlobalLoadStmt>(), true);

  EXPECT_EQ(block->statements[7]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[8]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[9]->is<GlobalLoadStmt>(), true);

  EXPECT_EQ(block->statements[10]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[11]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[12]->is<GlobalLoadStmt>(), true);
}

TEST(Scalarize, ScalarizeLocalStore) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = AllocaStmt()
    TensorType<4 x i32>  %2 = MatrixInitStmt([1, 1, 2, 2])
    StoreStmt(%1, %2)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  Stmt *dest_stmt = block->push_back<AllocaStmt>(tensor_type);
  dest_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  auto const_1_stmt = block->push_back<ConstStmt>(TypedConstant(1));
  auto const_2_stmt = block->push_back<ConstStmt>(TypedConstant(2));
  std::vector<Stmt *> matrix_init_vals = {const_1_stmt, const_1_stmt,
                                          const_2_stmt, const_2_stmt};
  auto matrix_init_stmt =
      block->push_back<MatrixInitStmt>(std::move(matrix_init_vals));
  matrix_init_stmt->ret_type = tensor_type;

  // LocalStoreStmt survives irpass::die()
  block->push_back<LocalStoreStmt>(dest_stmt, matrix_init_stmt);

  irpass::scalarize(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 2 /*const*/ + 4 /*alloca*/ + 4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[0]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[1]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<AllocaStmt>(), true);

  EXPECT_EQ(block->statements[4]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<ConstStmt>(), true);

  EXPECT_EQ(block->statements[6]->is<LocalStoreStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<LocalStoreStmt>(), true);
  EXPECT_EQ(block->statements[8]->is<LocalStoreStmt>(), true);
  EXPECT_EQ(block->statements[9]->is<LocalStoreStmt>(), true);
}

TEST(Scalarize, ScalarizeLocalLoad) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = AllocaStmt()
    LoadStmt(%1)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  Stmt *src_stmt = block->push_back<AllocaStmt>(tensor_type);
  src_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  auto load_stmt = block->push_back<LocalLoadStmt>(src_stmt);

  // Without this GlobalStoreStmt, nothing survives irpass::die()
  block->push_back<GlobalStoreStmt>(src_stmt, load_stmt);

  irpass::scalarize(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 4 /*alloca*/ + 4 /*load*/ + 4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[0]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[1]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<AllocaStmt>(), true);

  EXPECT_EQ(block->statements[4]->is<LocalLoadStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<LocalLoadStmt>(), true);
  EXPECT_EQ(block->statements[6]->is<LocalLoadStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<LocalLoadStmt>(), true);
}

TEST(Scalarize, ScalarizeBugTmp222) {
  // Test for Genesis bug
  // https://linear.app/genesis-ai-company/issue/CMP-151/fix-genesis-unit-test-bug-with-spirv-on-mac
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto one = block->push_back<ConstStmt>(TypedConstant(1));

  auto for1_body = std::make_unique<Block>();

  block->push_back<RangeForStmt>(zero, one, std::move(for1_body), false, 0, 0,
                                 false);

  auto for2_body = std::make_unique<Block>();
  for2_body->push_back<AllocaStmt>(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32));

  block->push_back<RangeForStmt>(zero, one, std::move(for2_body), false, 0, 0,
                                 false);

  auto root_snode = std::make_unique<SNode>(/*depth=*/0, /*t=*/SNodeType::root);
  const std::vector<Axis> axes = {Axis{0}};

  // create snodes
  auto dense_snode_ = &(root_snode->dense(axes, 1));
  auto leaf_snode = &(dense_snode_->insert_children(SNodeType::place));
  leaf_snode->dt = PrimitiveType::f32;
  auto leaf_snode2 = &(dense_snode_->insert_children(SNodeType::place));
  leaf_snode2->dt = PrimitiveType::f32;
  auto leaf_snode3 = &(dense_snode_->insert_children(SNodeType::place));
  leaf_snode3->dt = PrimitiveType::f32;
  auto leaf_snode4 = &(dense_snode_->insert_children(SNodeType::place));
  leaf_snode4->dt = PrimitiveType::f32;
  std::vector<SNode *> snodes = {leaf_snode, leaf_snode2, leaf_snode3,
                                 leaf_snode4};

  // create vector type and pointer to vector type
  std::vector<int> vector_shape = {4};
  auto vector_type = TypeFactory::get_instance().get_tensor_type(
      vector_shape,
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32));
  auto pointer_to_vector_type =
      TypeFactory::get_instance().get_pointer_type(vector_type);

  // create offloaded0
  block->push_back<OffloadedStmt>(OffloadedStmt::TaskType::range_for,
                                  Arch::vulkan, kernel.get());
  auto offloaded0 = block->statements.back()->as<OffloadedStmt>();
  offloaded0->body->push_back<ConstStmt>(TypedConstant(0));

  std::vector<Stmt *> indices0 = {zero};
  auto matrix_global_ptr0 = offloaded0->body->push_back<MatrixOfGlobalPtrStmt>(
      snodes, indices0, false, 1, pointer_to_vector_type, true);
  auto vector_alloc = offloaded0->body->push_back<AllocaStmt>(vector_type);
  // auto global_load0 =
  //     offloaded0->body->push_back<GlobalLoadStmt>(matrix_global_ptr0);
  // offloaded0->body->push_back<LocalStoreStmt>(pointer_vector_alloc,
  // global_load0);

  offloaded0->body->push_back<MatrixPtrStmt>(vector_alloc, zero);

  // create offloaded
  block->push_back<OffloadedStmt>(OffloadedStmt::TaskType::range_for,
                                  Arch::vulkan, kernel.get());
  auto offloaded = block->statements.back()->as<OffloadedStmt>();
  offloaded->body->push_back<ConstStmt>(TypedConstant(0));

  auto vector_alloca = offloaded->body->push_back<AllocaStmt>(vector_type);
  auto zero_for2 = offloaded->body->push_back<ConstStmt>(TypedConstant(0));
  std::vector<Stmt *> indices = {zero_for2};
  auto matrix_global_ptr = offloaded->body->push_back<MatrixOfGlobalPtrStmt>(
      snodes, indices, false, 1, pointer_to_vector_type, true);
  auto global_load =
      offloaded->body->push_back<GlobalLoadStmt>(matrix_global_ptr);
  // offloaded->body->push_back<LocalStoreStmt>(vector_alloca, global_load);
  // auto vector_alloca2 = offloaded->body->push_back<AllocaStmt>(vector_type);
  // auto vector_load2 =
  // offloaded->body->push_back<LocalLoadStmt>(vector_alloca);
  // offloaded->body->push_back<LocalStoreStmt>(vector_alloca2, vector_load2);
  auto matrix_ptr =
      offloaded->body->push_back<MatrixPtrStmt>(vector_alloca, zero_for2);
  offloaded->body->push_back<LocalLoadStmt>(matrix_ptr);

  irpass::print(block.get());

  irpass::scalarize(block.get());
  irpass::print(block.get());
}

}  // namespace gstaichi::lang
