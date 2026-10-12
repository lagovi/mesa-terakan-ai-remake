/*
 * Copyright © 2026 Terakan contributors
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"

bool terakan_nir_lower_subgroups(nir_shader * shader);

static bool
terakan_nir_lower_subgroups_filter(nir_instr const * const instr, void const * const cb_data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_load_num_subgroups: {
      /* These need the workgroup size, which only exists (non-variable) for compute. */
      nir_shader const * const shader = (nir_shader const *)cb_data;
      if (shader->info.stage != MESA_SHADER_COMPUTE || shader->info.workgroup_size_variable)
         return false;
      return true;
   }
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_elect:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_rotate:
   case nir_intrinsic_ballot:
   case nir_intrinsic_inverse_ballot:
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_ballot_bit_count_inclusive:
   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      return true;
   default:
      return false;
   }
}

static nir_def *
terakan_nir_singleton_ballot(nir_builder * const b, nir_def const * const def, nir_def * const bit)
{
   nir_def * components[NIR_MAX_VEC_COMPONENTS];
   components[0] = nir_b2iN(b, bit, def->bit_size);
   for (unsigned i = 1; i < def->num_components; ++i)
      components[i] = nir_imm_intN_t(b, 0, def->bit_size);
   return nir_vec(b, components, def->num_components);
}

static nir_def *
terakan_nir_singleton_ballot_bit(nir_builder * const b, nir_def * const ballot)
{
   return nir_i2b(b, nir_iand_imm(b, nir_channel(b, ballot, 0), 1));
}

static nir_def *
terakan_nir_subgroup_identity(nir_builder * const b, nir_intrinsic_instr const * const intrin)
{
   nir_op const reduction_op = (nir_op)nir_intrinsic_reduction_op(intrin);
   nir_const_value const identity = nir_alu_binop_identity(reduction_op, intrin->def.bit_size);
   /* One value per component: `nir_build_imm` reads as many as it is told there are, so handing it
    * a single one for a vector result reads past it. The identity is the same in every component.
    */
   nir_const_value identity_components[NIR_MAX_VEC_COMPONENTS];
   for (unsigned component = 0; component < intrin->def.num_components; ++component) {
      identity_components[component] = identity;
   }
   return nir_build_imm(b, intrin->def.num_components, intrin->def.bit_size, identity_components);
}

static nir_def *
terakan_nir_lower_subgroups_impl(nir_builder * const b, nir_instr * const instr,
                                 UNUSED void * const cb_data)
{
   nir_intrinsic_instr * const intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_subgroup_size:
      return nir_imm_int(b, 1);
   case nir_intrinsic_load_subgroup_id: {
      /* subgroupSize is 1, so every invocation forms its own subgroup, and the subgroup id
       * is the linear local invocation index. (The scalar index sysval itself is not
       * implemented by the backend, so compose it from the local invocation id vector.) */
      uint16_t const * const workgroup_size = b->shader->info.workgroup_size;
      nir_def * const local_id = nir_load_local_invocation_id(b);
      nir_def * subgroup_id = nir_channel(b, local_id, 0);
      subgroup_id = nir_iadd(
         b, subgroup_id,
         nir_imul_imm(b, nir_channel(b, local_id, 1), workgroup_size[0]));
      subgroup_id = nir_iadd(
         b, subgroup_id,
         nir_imul_imm(b, nir_channel(b, local_id, 2),
                      workgroup_size[0] * workgroup_size[1]));
      return subgroup_id;
   }
   case nir_intrinsic_load_num_subgroups: {
      /* subgroupSize is 1, so every invocation of the workgroup is its own subgroup. */
      uint16_t const * const workgroup_size = b->shader->info.workgroup_size;
      return nir_imm_int(b, workgroup_size[0] * workgroup_size[1] * workgroup_size[2]);
   }
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_first_invocation:
      return nir_imm_int(b, 0);

   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_le_mask:
      return terakan_nir_singleton_ballot(b, &intrin->def, nir_imm_true(b));
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
      return terakan_nir_singleton_ballot(b, &intrin->def, nir_imm_false(b));

   case nir_intrinsic_elect:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
      return nir_imm_true(b);

   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_rotate:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
      return intrin->src[0].ssa;

   case nir_intrinsic_ballot:
      return terakan_nir_singleton_ballot(b, &intrin->def, intrin->src[0].ssa);
   case nir_intrinsic_inverse_ballot:
      return terakan_nir_singleton_ballot_bit(b, intrin->src[0].ssa);
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_ballot_bit_count_inclusive:
      return nir_b2iN(b, terakan_nir_singleton_ballot_bit(b, intrin->src[0].ssa),
                      intrin->def.bit_size);
   case nir_intrinsic_ballot_bitfield_extract:
      return terakan_nir_singleton_ballot_bit(b, intrin->src[0].ssa);
   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
      return nir_imm_intN_t(b, 0, intrin->def.bit_size);

   case nir_intrinsic_exclusive_scan:
      return terakan_nir_subgroup_identity(b, intrin);

   default:
      unreachable("unexpected singleton subgroup intrinsic");
   }
}

bool
terakan_nir_lower_subgroups(nir_shader * const shader)
{
   return nir_shader_lower_instructions(shader, terakan_nir_lower_subgroups_filter,
                                        terakan_nir_lower_subgroups_impl, shader);
}
