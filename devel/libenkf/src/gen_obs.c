/**
   See the overview documentation of the observation system in
   enkf_obs.c
*/
#include <stdlib.h>
#include <enkf_util.h>
#include <enkf_types.h>
#include <enkf_macros.h>
#include <util.h>
#include <gen_obs.h>
#include <meas_vector.h>
#include <obs_data.h>
#include <gen_data.h>
#include <gen_obs.h>
#include <gen_common.h>
#include <gen_obs_active.h>

/**
   This file implemenets a structure for general observations. A
   general observation is just a vector of numbers - where EnKF has no
   understanding whatsover of the type of these data. The actual data
   is supposed to be found in a file.
   
   Currently it can only observe gen_data instances - but that should
   be generalized.
*/



#define GEN_OBS_TYPE_ID 77619

struct gen_obs_struct {
  int                          __type_id;
  int                          obs_size;        /* This is the total size of the observation vector. */ 
  int                        * data_index_list; /* The indexes which are observed in the corresponding gen_data instance - of length obs_size. */

  double                     * __obs_buffer;    /* This is the actual storage variable. obs_data and obs_std just point into this vector. */
  double                     * obs_data;        /* The observed data. */
  double                     * obs_std;         /* The observed standard deviation. */ 

  char                       * obs_file;        /* The file holding the observation. */ 
  gen_data_file_format_type    obs_format;      /* The format, i.e. ASCII, binary_double or binary_float, of the observation file. */
};

/******************************************************************/


void gen_obs_free(gen_obs_type * gen_obs) {
  util_safe_free(gen_obs->__obs_buffer);
  util_safe_free(gen_obs->obs_file);
  util_safe_free(gen_obs->data_index_list);
  free(gen_obs);
}



/**
   This function loads the actual observations from disk, and
   initializes the obs_data and obs_std pointers with the
   observations. It also sets the obs_size field of the gen_obs
   instance.

   The file with observations should be a long vector of 2N elements,
   where the first N elements are data values, and the last N values
   are the corresponding standard deviations.
   
   The file is loaded with the gen_common_fload_alloc() function, and
   can be in formatted ASCII or binary_float / binary_double. Observe
   that there is *NO* header information in this file.
*/


static void gen_obs_load_observation(gen_obs_type * gen_obs) {
  ecl_type_enum load_type;
  util_safe_free( gen_obs->__obs_buffer );
  
  gen_obs->obs_size = 0;
  gen_obs->__obs_buffer = gen_common_fload_alloc(gen_obs->obs_file , gen_obs->obs_format , ecl_double_type , &load_type , &gen_obs->obs_size);
  
  /** Ensure that the data is of type double. */
  if (load_type == ecl_float_type) {
    double * double_data = util_malloc(gen_obs->obs_size * sizeof * double_data , __func__);
    util_float_to_double(double_data , (const float *) gen_obs->__obs_buffer , gen_obs->obs_size);
    free(gen_obs->__obs_buffer);
    gen_obs->__obs_buffer = double_data;
  }
  
  gen_obs->obs_size /= 2; /* Originally contains BOTH data and std. */
  gen_obs->obs_data   =  gen_obs->__obs_buffer;
  gen_obs->obs_std    = &gen_obs->__obs_buffer[gen_obs->obs_size];
}





/**
   data_index_file is the name of a file with indices which should be
   observed, data_inde_string is the same, in the form of a
   "1,2,3,4-10, 17,19,22-100" string. Only one of these items can be
   != NULL. If both are NULL it is assumed that all the indices of the
   gen_data instance should be observed.
*/


gen_obs_type * gen_obs_alloc(const char * obs_file , const char * data_index_file , const char * data_index_string) {
  gen_obs_type * obs = util_malloc(sizeof * obs , __func__);
  
  obs->__type_id       = GEN_OBS_TYPE_ID;
  obs->__obs_buffer    = NULL;
  obs->obs_file        = util_alloc_string_copy( obs_file );
  obs->obs_format      = ASCII;  /* Hardcoded for now. */
  
  gen_obs_load_observation(obs); /* The observation data is loaded - and internalized at boot time - even though it might not be needed for a long time. */

  if ((data_index_file == NULL) && (data_index_string == NULL)) {
    /* 
       We observe all the elements in the remote (gen_data) instance,
       and the data_index_list just becomes a identity mapping. 
    */
    obs->data_index_list = util_malloc( obs->obs_size * sizeof * obs->data_index_list , __func__);
    for (int i =0; i < obs->obs_size; i++)
      obs->data_index_list[i] = i;
  } else {
    if (data_index_file != NULL) {
    } else {
    }
  }
  return obs;
}


/** Active - not active when it comes to local analysis is *NOT* handled. */
void gen_obs_measure(const gen_obs_type * gen_obs , const gen_data_type * gen_data , meas_vector_type * meas_vector) {
  int iobs;
  
  for (iobs = 0; iobs < gen_obs->obs_size; iobs++)
    meas_vector_add( meas_vector , gen_data_iget_double( gen_data , gen_obs->data_index_list[iobs] ));
  
}


double gen_obs_chi2(const gen_obs_type * gen_obs , const gen_data_type * gen_data) {
  int iobs;
  double sum_chi2 = 0;
  for (iobs = 0; iobs < gen_obs->obs_size; iobs++) {
    double x  = (gen_data_iget_double( gen_data , gen_obs->data_index_list[iobs]) - gen_obs->obs_data[iobs]) / gen_obs->obs_std[iobs];
    sum_chi2 += x*x;
  }
  return sum_chi2;
}



void gen_obs_get_observations(gen_obs_type * gen_obs , int report_step, obs_data_type * obs_data) {
  int iobs;
  const char * kw = "GEN_OBS";

  for (iobs = 0; iobs < gen_obs->obs_size; iobs++)
    obs_data_add( obs_data , gen_obs->obs_data[iobs] , gen_obs->obs_std[iobs] , kw);
}





void gen_obs_activate(gen_obs_type * obs , active_mode_type active_mode , void * __active) {
  //gen_obs_active_type * active = gen_obs_active_safe_cast(__active);
}



void gen_obs_user_get(const gen_obs_type * gen_obs , const char * index_key , double * value , double * std , bool * valid) {
  *valid = true;
  *value = 1.0;
  *std   = 1.0;
}


  
/*****************************************************************/
SAFE_CAST(gen_obs , GEN_OBS_TYPE_ID)
IS_INSTANCE(gen_obs , GEN_OBS_TYPE_ID)
VOID_OBS_ACTIVATE(gen_obs)
VOID_FREE(gen_obs)
VOID_GET_OBS(gen_obs)
VOID_MEASURE(gen_obs , gen_data)
VOID_USER_GET_OBS(gen_obs)
VOID_CHI2(gen_obs , gen_data)
