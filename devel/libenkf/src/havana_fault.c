#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <enkf_types.h>
#include <fortio.h>
#include <util.h>
#include <havana_fault_config.h>
#include <havana_fault.h>
#include <enkf_util.h>
#include <math.h>
#include <scalar.h>
#include <assert.h>
#include <subst.h>



GET_DATA_SIZE_HEADER(havana_fault);


struct havana_fault_struct 
{
  int                             __type_id; 
  const havana_fault_config_type *config;
  scalar_type                    *scalar;
};

/*****************************************************************/

void havana_fault_free_data(havana_fault_type *havana_fault) {
  assert(havana_fault->scalar);
  scalar_free_data(havana_fault->scalar);
}



void havana_fault_free(havana_fault_type *havana_fault) {
  scalar_free(havana_fault->scalar);
  free(havana_fault);
}



void havana_fault_realloc_data(havana_fault_type *havana_fault) {
  scalar_realloc_data(havana_fault->scalar);
}


void havana_fault_output_transform(const havana_fault_type * havana_fault) {
  scalar_transform(havana_fault->scalar);
}

void havana_fault_set_data(havana_fault_type * havana_fault , const double * data) {
  scalar_set_data(havana_fault->scalar , data);
}


void havana_fault_get_data(const havana_fault_type * havana_fault , double * data) {
  scalar_get_data(havana_fault->scalar , data);
}

void havana_fault_get_output_data(const havana_fault_type * havana_fault , double * output_data) {
  scalar_get_output_data(havana_fault->scalar , output_data);
}


const double * havana_fault_get_data_ref(const havana_fault_type * havana_fault) {
  return scalar_get_data_ref(havana_fault->scalar);
}


const double * havana_fault_get_output_ref(const havana_fault_type * havana_fault) {
  return scalar_get_output_ref(havana_fault->scalar);
}



havana_fault_type * havana_fault_alloc(const havana_fault_config_type * config) {
  havana_fault_type * havana_fault  = util_malloc(sizeof * havana_fault , __func__);
  havana_fault->config = config;
  gen_kw_config_type * gen_kw_config = havana_fault_config_get_gen_kw_config( config );
  havana_fault->scalar               = scalar_alloc(gen_kw_config_get_scalar_config( gen_kw_config )); 
  havana_fault->__type_id = HAVANA_FAULT;
  return havana_fault;
}



void havana_fault_clear(havana_fault_type * havana_fault) {
  scalar_clear(havana_fault->scalar);
}



havana_fault_type * havana_fault_copyc(const havana_fault_type *havana_fault) {
  havana_fault_type * new = havana_fault_alloc(havana_fault->config); 
  scalar_memcpy(new->scalar , havana_fault->scalar);
  return new; 
}


bool havana_fault_fwrite(const havana_fault_type *havana_fault , FILE * stream , bool internal_state) {
  enkf_util_fwrite_target_type(stream , HAVANA_FAULT);
  scalar_stream_fwrite(havana_fault->scalar , stream , internal_state);
  return true;
}


void havana_fault_fread(havana_fault_type * havana_fault , FILE * stream) {
  enkf_util_fread_assert_target_type(stream , HAVANA_FAULT);
  scalar_stream_fread(havana_fault->scalar , stream);
}



void havana_fault_truncate(havana_fault_type * havana_fault) {
  scalar_truncate( havana_fault->scalar );  
}


bool  havana_fault_initialize(havana_fault_type *havana_fault, int iens) { 
  scalar_sample(havana_fault->scalar);
  return true;
} 


int havana_fault_serialize(const havana_fault_type *havana_fault , serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  return scalar_serialize(havana_fault->scalar , serial_state , serial_offset , serial_vector);
}

void havana_fault_deserialize(havana_fault_type *havana_fault , serial_state_type * serial_state, const serial_vector_type * serial_vector) {
  scalar_deserialize(havana_fault->scalar , serial_state , serial_vector);
}




havana_fault_type * havana_fault_alloc_mean(int ens_size , const havana_fault_type **havana_fault_ens) {
  int iens;
  havana_fault_type * avg_havana_fault = havana_fault_copyc(havana_fault_ens[0]);
  for (iens = 1; iens < ens_size; iens++) 
    havana_fault_iadd(avg_havana_fault , havana_fault_ens[iens]);
  havana_fault_iscale(avg_havana_fault , 1.0 / ens_size);
  return avg_havana_fault;
}


void havana_fault_filter_file(const havana_fault_type * havana_fault , const char * run_path, int *ntarget_ref, char ***target_ref) 
{
  const int size             = havana_fault_config_get_data_size(havana_fault->config);
  const double * output_data = scalar_get_output_ref(havana_fault->scalar);
  subst_list_type * subst_list = subst_list_alloc();
  int ikw;
  int ntemplates = 0;
  char ** target;
  
  havana_fault_output_transform(havana_fault);
  for (ikw = 0; ikw < size; ikw++) {
    char * tagged_fault = enkf_util_alloc_tagged_string( havana_fault_config_get_name(havana_fault->config , ikw) );
    subst_list_insert_owned_ref( subst_list , tagged_fault , util_alloc_sprintf( "%g" , output_data[ikw] ));
    free( tagged_fault );
  }
  

  /* 
     Scan through the list of template files and create target files. 
  */
 {
   const char *template_file_list = havana_fault_config_get_template_ref(havana_fault->config);
   FILE * stream = util_fopen(template_file_list,"r");
   bool end_of_file;
   fscanf(stream,"%d",&ntemplates);
   printf("%s %d\n","Number of template files: ",ntemplates);
   
   target             = util_malloc(ntemplates * sizeof(char *) , __func__);
   
   for( int i=0; i < ntemplates; i++) {
       char * target_file_root;
       char * template_file;
       
       util_forward_line(stream,&end_of_file);
       if(end_of_file) 
	util_abort("%s: Premature end of file when reading list of template files for Havana from:%s \n",__func__ , template_file_list);
      
      /* Read template file */
      template_file =  util_fscanf_alloc_token(stream);
      /* printf("%s\n",template_file); */

      /* Read target file root */
      target_file_root = util_fscanf_alloc_token(stream);
      /* printf("%s\n",target_file_root); */

      target[i] = util_alloc_filename(run_path , target_file_root , NULL);


      printf("%s   %s  \n",template_file,target[i]); 
      
      subst_list_filter_file(subst_list , template_file , target[i]);
      free(target_file_root);
      free(template_file);
   }
   fclose(stream);
 }
  
  
  /* Return values */
  *ntarget_ref     = ntemplates;
  *target_ref      = target;
  subst_list_free(subst_list);
}



#define PRINT_LINE(n,c,stream) { int _i; for (_i = 0; _i < (n); _i++) fputc(c , stream); fprintf(stream,"\n"); }
void havana_fault_ensemble_fprintf_results(const havana_fault_type ** ensemble, int ens_size , const char * filename) {
  const havana_fault_config_type * config = ensemble[0]->config;
  const int float_width     =  9;
  const int float_precision =  5;
  int        size    = havana_fault_config_get_data_size(ensemble[0]->config);
  int      * width   = util_malloc((size + 1) * sizeof * width , __func__);
  int        ikw , total_width;

  havana_fault_type * mean;
  havana_fault_type * std;

  havana_fault_alloc_stats(ensemble , ens_size , &mean , &std);
  width[0] = strlen("Member #|");
  total_width = width[0];
  for (ikw = 0; ikw < size; ikw++) {
    width[ikw + 1]  = util_int_max(strlen(havana_fault_config_get_name(config , ikw)), 2 * float_width + 5) + 1;  /* Must accomodate A +/- B */
    width[ikw + 1] += ( 1 - (width[ikw + 1] & 1)); /* Ensure odd length */
    total_width += width[ikw + 1] + 1;
  }
  
  {
    FILE * stream = util_fopen(filename , "w");
    int iens;

    util_fprintf_string("Member #|" , width[0] , true , stream);
    for (ikw = 0; ikw < size; ikw++) {
      util_fprintf_string(havana_fault_config_get_name(config , ikw) , width[ikw + 1] , center , stream);
      fprintf(stream , "|");
    }
    fprintf(stream , "\n");
    PRINT_LINE(total_width , '=' , stream);

    util_fprintf_string("Mean" , width[0] - 1 , true , stream);
    fprintf(stream , "|");
    {
      const double * mean_data = scalar_get_output_ref(mean->scalar);
      const double * std_data  = scalar_get_output_ref(std->scalar);
      for (ikw = 0; ikw < size; ikw++) {
	int w = (width[ikw + 1] - 5) / 2;
	util_fprintf_double(mean_data[ikw] , w , float_precision , 'g' , stream);
	fprintf(stream , " +/- ");
	util_fprintf_double(std_data[ikw] , w , float_precision , 'g' , stream);
	fprintf(stream , "|");
      }
      fprintf(stream , "\n");
    }
    PRINT_LINE(total_width , '-' , stream);
    for (iens = 0; iens < ens_size; iens++) {
      const double * data = scalar_get_output_ref(ensemble[iens]->scalar);
      util_fprintf_int(iens + 1, width[0] - 1 , stream);
      fprintf(stream , "|");
      
      for (ikw = 0; ikw < size; ikw++) {
	util_fprintf_double(data[ikw] , width[ikw + 1] , float_precision , 'g' , stream);
	fprintf(stream , "|");
      }
      fprintf(stream , "\n");
    }
    PRINT_LINE(total_width , '=' , stream);
    fclose(stream);
  }
  
  havana_fault_free(mean);
  havana_fault_free(std);
  free(width);
}
#undef PRINT_LINE





/**
  This function writes the results for eclipse to use. Observe that
  for this function the second argument is a target_path (the
  config_object has been allocated with target_file == NULL).

*/


void havana_fault_ecl_write(const havana_fault_type * havana_fault , const char * run_path , const char * file /* This is NOT used. */ , fortio_type * fortio) {
  havana_fault_config_run_havana(havana_fault->config , havana_fault->scalar ,  run_path);
}


void havana_fault_export(const havana_fault_type * havana_fault , int * _size , char ***_kw_list , double **_output_values) {
  havana_fault_output_transform(havana_fault);

  *_kw_list       = havana_fault_config_get_name_list(havana_fault->config);
  *_size          = havana_fault_config_get_data_size(havana_fault->config);
  *_output_values = (double *) scalar_get_output_ref(havana_fault->scalar);

}

const char * havana_fault_get_name(const havana_fault_type * havana_fault, int kw_nr) {
  return  havana_fault_config_get_name(havana_fault->config , kw_nr);
}



double havana_fault_user_get(const havana_fault_type * havana_fault , const char * index_string , bool * valid) {
  const bool internal_value = false;
  gen_kw_config_type * gen_kw_config = havana_fault_config_get_gen_kw_config( havana_fault->config );
  int index                          = gen_kw_config_get_index( gen_kw_config , index_string );
  if (index < 0) {
    *valid = false;
    return 0;
  } else {
    *valid = true;
    return scalar_iget_double(havana_fault->scalar , internal_value , index);
  }
}




/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

SAFE_CAST(havana_fault , HAVANA_FAULT);
MATH_OPS_SCALAR(havana_fault);
ALLOC_STATS(havana_fault);
VOID_USER_GET(havana_fault);
VOID_ALLOC(havana_fault);
VOID_REALLOC_DATA(havana_fault);
VOID_SERIALIZE (havana_fault);
VOID_DESERIALIZE (havana_fault);
VOID_INITIALIZE(havana_fault);
VOID_FREE_DATA(havana_fault)
VOID_FWRITE (havana_fault)
VOID_FREAD  (havana_fault)
VOID_COPYC  (havana_fault)
VOID_FREE   (havana_fault)
VOID_ECL_WRITE(havana_fault)
VOID_FPRINTF_RESULTS(havana_fault)



