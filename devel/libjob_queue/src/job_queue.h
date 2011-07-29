/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'job_queue.h' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#ifndef __JOB_QUEUE_H__
#define __JOB_QUEUE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <pthread.h>
#include <queue_driver.h>
#include <path_fmt.h>

typedef struct job_queue_struct job_queue_type;
void                job_queue_submit_complete( job_queue_type * queue );
job_driver_type     job_queue_get_driver_type( const job_queue_type * queue );
void                job_queue_set_driver(job_queue_type * queue , queue_driver_type * driver);
  //void                job_queue_set_size( job_queue_type * job_queue , int size );
void                job_queue_set_runpath_fmt(job_queue_type *  , const path_fmt_type * );
job_queue_type   *  job_queue_alloc( int  , bool , const char * ok_file , const char * exit_file);
void                job_queue_free(job_queue_type *);
int                 job_queue_add_job_mt(job_queue_type * , const char * run_cmd , int num_cpu , const char * , const char * , int argc , const char ** argv );
int                 job_queue_add_job_st(job_queue_type * , const char * run_cmd , int num_cpu , const char * , const char * , int argc , const char ** argv );
void                job_queue_run_jobs(job_queue_type * , int , bool verbose);
void                job_queue_run_jobs_threaded(job_queue_type * queue , int num_total_run, bool verbose);
void *              job_queue_run_jobs__(void * );
job_status_type     job_queue_get_job_status(job_queue_type * , int );
void                job_queue_set_load_OK(job_queue_type * queue , int job_index);
void                job_queue_set_all_fail(job_queue_type * queue , int job_index);
void                job_queue_set_external_restart(job_queue_type * queue , int job_index);
void                job_queue_set_external_fail(job_queue_type * queue , int job_index);
void                job_queue_set_external_load(job_queue_type * queue , int job_index);
const char        * job_queue_status_name( job_status_type status );
void                job_queue_set_max_running( job_queue_type * queue , int max_running );
int                 job_queue_inc_max_runnning( job_queue_type * queue, int delta );
int                 job_queue_get_max_running( const job_queue_type * queue );
int                 job_queue_iget_status_summary( const job_queue_type * queue , job_status_type status);
time_t              job_queue_iget_sim_start( job_queue_type * queue, int job_index);
time_t              job_queue_iget_submit_time( job_queue_type * queue, int job_index);
job_driver_type     job_queue_lookup_driver_name( const char * driver_name );

bool                job_queue_kill_job( job_queue_type * queue , int job_index);
bool                job_queue_is_running( const job_queue_type * queue );
void                job_queue_set_max_submit( job_queue_type * job_queue , int max_submit );
int                 job_queue_get_max_submit(const job_queue_type * job_queue );

bool                job_queue_get_pause( const job_queue_type * job_queue );
void                job_queue_set_pause_on( job_queue_type * job_queue);
void                job_queue_set_pause_off( job_queue_type * job_queue);
void                job_queue_user_exit( job_queue_type * queue);
void              * job_queue_iget_job_data( job_queue_type * job_queue , int job_nr );

int                 job_queue_get_num_running( const job_queue_type * queue);
int                 job_queue_get_num_pending( const job_queue_type * queue);
int                 job_queue_get_num_waiting( const job_queue_type * queue);
int                 job_queue_get_num_complete( const job_queue_type * queue);
int                 job_queue_get_num_failed( const job_queue_type * queue);



#ifdef __cplusplus
}
#endif
#endif

