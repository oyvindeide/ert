#ifndef ERT_GEN_DATA_CONFIG_H
#define ERT_GEN_DATA_CONFIG_H
#include <stdbool.h>

#include <ert/util/bool_vector.h>
#include <ert/util/stringlist.h>
#include <ert/util/util.h>

#include <ert/enkf/enkf_fs_type.hpp>
#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_types.hpp>

typedef enum {
    GEN_DATA_UNDEFINED = 0,
    /** The file is ASCII file with a vector of numbers formatted with "%g".*/
    ASCII = 1,
    /** The data is inserted into a user defined template file. */
    ASCII_TEMPLATE = 2,
} /*   The data is in a binary file with floats. */
gen_data_file_format_type;

typedef struct gen_data_config_struct gen_data_config_type;

void gen_data_config_load_active(gen_data_config_type *config, enkf_fs_type *fs,
                                 int report_step, bool force_load);
bool gen_data_config_valid_result_format(const char *result_file_fmt);

bool gen_data_config_has_active_mask(const gen_data_config_type *config,
                                     enkf_fs_type *fs, int report_step);

extern "C" gen_data_config_type *
gen_data_config_alloc_GEN_DATA_result(const char *key,
                                      gen_data_file_format_type input_format);
void gen_data_config_set_ens_size(gen_data_config_type *config, int ens_size);
extern "C" gen_data_file_format_type
gen_data_config_get_input_format(const gen_data_config_type *);
extern "C" void gen_data_config_free(gen_data_config_type *);
extern "C" PY_USED int
gen_data_config_get_initial_size(const gen_data_config_type *config);
void gen_data_config_assert_size(gen_data_config_type *, int, int);
extern "C" const bool_vector_type *
gen_data_config_get_active_mask(const gen_data_config_type *config);
void gen_data_config_update_active(gen_data_config_type *config,
                                   int report_step,
                                   const bool_vector_type *data_mask,
                                   enkf_fs_type *sim_fs);
extern "C" const char *
gen_data_config_get_key(const gen_data_config_type *config);
int gen_data_config_get_data_size(const gen_data_config_type *config,
                                  int report_step);
gen_data_file_format_type
gen_data_config_check_format(const char *format_string);

const int_vector_type *
gen_data_config_get_active_report_steps(const gen_data_config_type *config);
extern "C" int
gen_data_config_iget_report_step(const gen_data_config_type *config, int index);
void gen_data_config_add_report_step(gen_data_config_type *config,
                                     int report_step);
extern "C" bool
gen_data_config_has_report_step(const gen_data_config_type *config,
                                int report_step);
extern "C" int
gen_data_config_num_report_step(const gen_data_config_type *config);
extern "C" int
gen_data_config_get_data_size__(const gen_data_config_type *config,
                                int report_step);

VOID_FREE_HEADER(gen_data_config)

#endif
