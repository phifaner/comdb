/*! \file interface.cu
 *  \brief Implementation for interface declared in interface.h
 */

#include "interface.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <string>
#include <sys/time.h>
#include <ctime>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>

#include <thrust/system_error.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Logger.h"
#include "Timing.h"

using namespace GPUGenie;
using namespace std;


//merge result
void merge_knn_results_from_multiload(std::vector<std::vector<int> >& _result,
		std::vector<std::vector<int> >& _result_count, std::vector<int>& result,
		std::vector<int>& result_count, int table_num, int topk, int query_num)
{
	for (int i = 0; i < query_num; ++i)
	{
		vector<int> _sort;
		vector<int> _sort_count;
		for (int j = 0; j < table_num; ++j)
		{
			_sort.insert(_sort.end(), _result[j].begin() + i * topk,
					_result[j].begin() + (i + 1) * topk);
			_sort_count.insert(_sort_count.end(),
					_result_count[j].begin() + i * topk,
					_result_count[j].begin() + (i + 1) * topk);
		}

		for (int j = 0; j < topk; ++j)
		{
			if (_sort_count.begin() == _sort_count.end())
			{
				throw GPUGenie::cpu_runtime_error("No result!");
			}
			unsigned int index;
			index = std::distance(_sort_count.begin(),
					std::max_element(_sort_count.begin(), _sort_count.end()));

			result.push_back(_sort[index]);
			result_count.push_back(_sort_count[index]);
			_sort.erase(_sort.begin() + index);
			_sort_count.erase(_sort_count.begin() + index);

		}
	}
}
bool GPUGenie::preprocess_for_knn_csv(GPUGenie_Config& config,
		inv_table * &_table)
{
	unsigned int cycle = 0;

	if (config.max_data_size >= config.data_points->size() || config.max_data_size <= 0)
	{
		if (config.data_points->size() > 0)
		{
            		_table = new inv_table[1];
            		_table[0].set_table_index(0);
            		_table[0].set_total_num_of_table(1);
			Logger::log(Logger::DEBUG, "build from data_points...");
			switch (config.search_type)
			{
				case 0:
					load_table(_table[0], *(config.data_points), config);
					break;
				case 1:
					load_table_bijectMap(_table[0], *(config.data_points), config);
					break;
            	case 2:
                	load_table_sequence(_table[0], *(config.data_points), config);
                	break;
				default:
					throw GPUGenie::cpu_runtime_error("Unrecognised search type!");
			}
		}
		else
		{
			throw GPUGenie::cpu_runtime_error("no data input!");
		}
	}
	else
	{
		Logger::log(Logger::DEBUG, "build from data_points...");
        	unsigned int table_num;
		if (config.data_points->size() % config.max_data_size == 0)
		{
			table_num = config.data_points->size() / config.max_data_size;
			cycle = table_num;
		}
		else
		{
			table_num = config.data_points->size() / config.max_data_size + 1;
			cycle = table_num - 2;
		}

		_table = new inv_table[table_num];

		for (unsigned int i = 0; i < cycle; ++i)
		{
			vector<vector<int> > temp;
			temp.insert(temp.end(),
					config.data_points->begin() + i * config.max_data_size,
					config.data_points->begin()
							+ (i + 1) * config.max_data_size);

            		_table[i].set_table_index(i);
            		_table[i].set_total_num_of_table(table_num);
			switch (config.search_type)
			{
				case 0:
					load_table(_table[i], temp, config);
					break;
				case 1:
					load_table_bijectMap(_table[i], temp, config);
					break;
            	case 2:
                	load_table_sequence(_table[i], temp, config);
               		break;
				default:
					throw GPUGenie::cpu_runtime_error("Unrecognised search type!");
			}
		}
		if (table_num != cycle)
		{
			vector<vector<int> > temp1;
			vector<vector<int> > temp2;
			unsigned int second_last_size = (config.data_points->size()
					- cycle * config.max_data_size) / 2;
			temp1.insert(temp1.end(),
					config.data_points->begin() + cycle * config.max_data_size,
					config.data_points->begin() + cycle * config.max_data_size
					+ second_last_size);
			temp2.insert(temp2.end(),
					config.data_points->begin() + cycle * config.max_data_size
					+ second_last_size, config.data_points->end());

            		_table[cycle].set_table_index(cycle);
           	    	_table[cycle].set_total_num_of_table(table_num);
           		_table[cycle + 1].set_table_index(cycle+1);
           	 	_table[cycle + 1].set_total_num_of_table(table_num);
			switch (config.search_type)
			{
				case 0:
					load_table(_table[cycle], temp1, config);
					load_table(_table[cycle + 1], temp2, config);
					break;
				case 1:
					load_table_bijectMap(_table[cycle], temp1, config);
					load_table_bijectMap(_table[cycle + 1], temp2, config);
			        break;
            	case 2:
                	load_table_sequence(_table[cycle], temp1, config);
               		load_table_sequence(_table[cycle + 1], temp2,config);
                	break;
				default:
					throw GPUGenie::cpu_runtime_error("Unrecognised search type!");
			}
		}
	}
	return true;
}

bool GPUGenie::preprocess_for_knn_binary(GPUGenie_Config& config,
		inv_table * &_table)
{
	unsigned int cycle = 0;
	if (config.max_data_size >= config.row_num || config.max_data_size <= 0)
	{
		if (config.item_num != 0 && config.index != NULL && config.item_num != 0 && config.row_num != 0)
		{
            		_table = new inv_table[1];
            		_table[0].set_table_index(0);
            		_table[0].set_total_num_of_table(1);
			Logger::log(Logger::DEBUG, "build from data array...");
			switch (config.search_type)
			{
				case 0:
					load_table(_table[0], config.data, config.item_num, config.index,
						config.row_num, config);
					break;
				case 1:
					load_table_bijectMap(_table[0], config.data, config.item_num,
						config.index, config.row_num, config);
					break;
        			case 2:
					//binary reading is gradually deprecated
                			break;
			}
		}
		else
		{
			throw GPUGenie::cpu_runtime_error("no data input!");
		}
	}
	else
	{
		Logger::log(Logger::DEBUG, "build from data array...");
       		unsigned int table_num;
		if (config.row_num % config.max_data_size == 0)
		{
			table_num = config.row_num / config.max_data_size;
			cycle = table_num;
		}
		else
		{
			table_num = config.row_num / config.max_data_size + 1;
			cycle = table_num - 2;
		}

		_table = new inv_table[table_num];
		for (unsigned int i = 0; i < cycle; ++i)
		{
			unsigned int item_num = 0;
			item_num = config.index[(i + 1) * config.max_data_size]
					- config.index[i * config.max_data_size];
			if (i == table_num - 1)
				item_num = config.item_num
						- config.index[config.max_data_size * (table_num - 1)];
            		_table[i].set_table_index(i);
            		_table[i].set_total_num_of_table(table_num);
			switch (config.search_type)
			{
				case 0:
					load_table(_table[i],
						config.data + config.index[config.max_data_size * i],
						item_num, config.index + config.max_data_size * i,
						config.max_data_size, config);
					break;
				case 1:
					load_table_bijectMap(_table[i],
						config.data + config.index[config.max_data_size * i],
						item_num, config.index + config.max_data_size * i,
						config.max_data_size, config);
					break;
            	case 2:
					//binary reading is deprecated
                	break;
			}
		}

		if (table_num != cycle)
		{
			unsigned int second_last_row_size = (config.row_num
					- cycle * config.max_data_size) / 2;
			unsigned int last_row_size = config.row_num - second_last_row_size
					- cycle * config.max_data_size;
			unsigned int second_last_item_size =
					config.index[config.max_data_size * cycle
							+ second_last_row_size]
							- config.index[config.max_data_size * cycle];
			unsigned int last_item_size = config.item_num
					- config.index[config.max_data_size * cycle
							+ second_last_row_size];
            		_table[cycle].set_table_index(cycle);
            		_table[cycle].set_total_num_of_table(table_num);
            		_table[cycle + 1].set_table_index(cycle+1);
            		_table[cycle + 1].set_total_num_of_table(table_num);
			switch (config.search_type)
			{
				case 0:
					load_table(_table[cycle],
						config.data + config.index[config.max_data_size * cycle],
						second_last_item_size,
						config.index + config.max_data_size * cycle,
						second_last_row_size, config);
					load_table(_table[cycle + 1],
						config.data + config.index[config.max_data_size * cycle
										+ second_last_row_size], last_item_size,
						config.index + config.max_data_size * cycle
								+ second_last_row_size, last_row_size, config);
					break;
				case 1:
					load_table_bijectMap(_table[cycle],
						config.data + config.index[config.max_data_size * cycle],
						second_last_item_size,
						config.index + config.max_data_size * cycle,
						second_last_row_size, config);
					load_table_bijectMap(_table[cycle + 1],
						config.data
								+ config.index[config.max_data_size * cycle
										+ second_last_row_size], last_item_size,
						config.index + config.max_data_size * cycle
								+ second_last_row_size, last_row_size, config);
					break;
            	case 2:
					//deprecated
            		break;
			}
		}
	}
	return true;
}

void GPUGenie::knn_search_after_preprocess(GPUGenie_Config& config,
		inv_table * &_table, std::vector<int>& result,
		std::vector<int>& result_count)
{
    std::vector<query> queries;
    vector<vector<int> > _result;
    vector<vector<int> > _result_count;
    unsigned int table_num = _table[0].get_total_num_of_table();
    _result.resize(table_num);
    _result_count.resize(table_num);

    unsigned int accumulate_num = 0;
    for (unsigned int i = 0; i < table_num; ++i)
    {
        queries.clear();
        load_query(_table[i], queries, config);

        knn_search(_table[i], queries, _result[i], _result_count[i],config);

        if (i <= 0) continue;
        accumulate_num += _table[i - 1].i_size();
        for (unsigned int j = 0; j < _result[i].size(); ++j)
        {
            _result[i][j] += accumulate_num;
        }
    }

    merge_knn_results_from_multiload(_result, _result_count, result,
            result_count, table_num, config.num_of_topk, queries.size());


}
void GPUGenie::load_table(inv_table& table,
		std::vector<std::vector<int> >& data_points, GPUGenie_Config& config)
{
	inv_list list;
	u32 i, j;

	Logger::log(Logger::DEBUG, "Data row size: %d. Data Row Number: %d.",
			data_points[0].size(), data_points.size());
	u64 starttime = getTime();

	for (i = 0; i < data_points[0].size(); ++i)
	{
		std::vector<int> col;
		col.reserve(data_points.size());
		for (j = 0; j < data_points.size(); ++j)
		{
			col.push_back(data_points[j][i]);
		}
		list.invert(col);
		table.append(list);
	}

	table.build(config.posting_list_max_length, config.use_load_balance);

	if (config.save_to_gpu)
		table.cpy_data_to_gpu();
	table.is_stored_in_gpu = config.save_to_gpu;

	u64 endtime = getTime();
	double timeInterval = getInterval(starttime, endtime);
	Logger::log(Logger::DEBUG,
			"Before finishing loading. i_size():%d, m_size():%d.",
			table.i_size(), table.m_size());
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: loading index takes %f ms<<<<",
			timeInterval);

}

void GPUGenie::load_table(inv_table& table, int *data, unsigned int item_num,
		unsigned int *index, unsigned int row_num, GPUGenie_Config& config)
{
	inv_list list;
	u32 i, j;

	unsigned int row_size;
	unsigned int index_start_pos = 0;
	if (row_num == 1)
		row_size = item_num;
	else
		row_size = index[1] - index[0];

	index_start_pos = index[0];

	Logger::log(Logger::DEBUG, "Data row size: %d. Data Row Number: %d.",
			index[1], row_num);
	u64 starttime = getTime();

	for (i = 0; i < row_size; ++i)
	{
		std::vector<int> col;
		col.reserve(row_num);
		for (j = 0; j < row_num; ++j)
		{
			col.push_back(data[index[j] + i - index_start_pos]);
		}
		list.invert(col);
		table.append(list);
	}

	table.build(config.posting_list_max_length, config.use_load_balance);

	if (config.save_to_gpu)
		table.cpy_data_to_gpu();
	table.is_stored_in_gpu = config.save_to_gpu;

	u64 endtime = getTime();
	double timeInterval = getInterval(starttime, endtime);
	Logger::log(Logger::DEBUG,
			"Before finishing loading. i_size() : %d, m_size() : %d.",
			table.i_size(), table.m_size());
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: loading index takes %f ms<<<<",
			timeInterval);

}

void GPUGenie::load_query(inv_table& table, std::vector<query>& queries,
		GPUGenie_Config& config)
{
    if(config.search_type == 2)
    {
        load_query_sequence(table, queries, config);
        return;
    }
	if (config.use_multirange)
	{
		load_query_multirange(table, queries, config);
	}
	else
	{
		load_query_singlerange(table, queries, config);
	}
}

//Read new format query data
//Sample data format
//qid dim value selectivity weight
// 0   0   15     0.04        1
// 0   1   6      0.04        1
// ....
void GPUGenie::load_query_multirange(inv_table& table,
		std::vector<query>& queries, GPUGenie_Config& config)
{
	queries.clear();
	map<int, query> query_map;
	int qid, dim, val;
	float sel, weight;
	for (unsigned int iq = 0; iq < config.multirange_query_points->size(); ++iq)
	{
		attr_t& attr = (*config.multirange_query_points)[iq];

		qid = attr.qid;
		dim = attr.dim;
		val = attr.value;
		weight = attr.weight;
		sel = attr.sel;
		if (query_map.find(qid) == query_map.end())
		{
			query q(table, qid);
			q.topk(config.num_of_topk);
			if (config.selectivity > 0.0f)
			{
				q.selectivity(config.selectivity);
			}
			if (config.use_load_balance)
			{
				q.use_load_balance = true;
			}
			query_map[qid] = q;

		}
		query_map[qid].attr(dim, val, weight, sel, query_map[qid].count_ranges());
	}
	for (std::map<int, query>::iterator it = query_map.begin();
			it != query_map.end() && queries.size() < (unsigned int) config.num_of_queries;
			++it)
	{
		query& q = it->second;
		q.apply_adaptive_query_range();
		queries.push_back(q);
	}

	Logger::log(Logger::INFO, "Finish loading queries!");
	Logger::log(Logger::DEBUG, "%d queries are loaded.", queries.size());

}
void GPUGenie::load_query_singlerange(inv_table& table,
		std::vector<query>& queries, GPUGenie_Config& config)
{

	Logger::log(Logger::DEBUG, "Table dim: %d.", table.m_size());
	u64 starttime = getTime();

	u32 i, j;
	int value;
	int radius = config.query_radius;
	std::vector<std::vector<int> >& query_points = *config.query_points;
	for (i = 0; i < query_points.size(); ++i)
	{
		query q(table, i);

		for (j = 0;
				j < query_points[i].size()
						&& (config.search_type == 1 || j < (unsigned int) config.dim); ++j)
		{
			value = query_points[i][j];

			q.attr(config.search_type == 1 ? 0 : j,
					value - radius, value + radius,
					GPUGENIE_DEFAULT_WEIGHT, j);
		}

		q.topk(config.num_of_topk);
		q.selectivity(config.selectivity);
		if (config.use_adaptive_range)
		{
			q.apply_adaptive_query_range();
		}
		if (config.use_load_balance)
		{
			q.use_load_balance = true;
		}

		queries.push_back(q);
	}

	u64 endtime = getTime();
	double timeInterval = getInterval(starttime, endtime);
	Logger::log(Logger::INFO, "%d queries are created!", queries.size());
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: loading query takes %f ms<<<<",
			timeInterval);
}

void GPUGenie::load_query_sequence(inv_table& table,
		vector<query>& queries, GPUGenie_Config& config)
{

	Logger::log(Logger::DEBUG, "Table dim: %d.", table.m_size());
	u64 starttime = getTime();

	u32 i, j;
	int value, min_value;
    	min_value = table.get_min_value_sequence();

	vector<vector<int> >& query_points = *config.query_points;
	vector<vector<int> > converted_query;
    	for(i = 0 ; i<query_points.size() ; ++i)
    	{
        	vector<int> line;
        	for(j = 0 ; j<query_points[i].size() ; ++j)
            		line.push_back(query_points[i][j] - min_value);
        	converted_query.push_back(line);
    	}

    	vector<vector<int> > _gram_query, gram_query;
    	sequence_to_gram(converted_query, _gram_query, table.get_max_value_sequence() - min_value, table.get_gram_length_sequence());

    	unordered_map<int, int> _map;
    
    	for(i = 0; i < _gram_query.size(); ++i)
    	{
        	vector<int> line;
        	for(j = 0; j < _gram_query[i].size(); ++j)
        	{
            		unordered_map<int, int>::iterator result = _map.find(_gram_query[i][j]);
            		if(result == _map.end())
            		{
                		_map.insert({_gram_query[i][j], 0});
                		line.push_back(_gram_query[i][j]<<table.shift_bits_sequence);
            		}
            		else
            		{
                		result->second += 1;
                		line.push_back((result->first<<table.shift_bits_sequence) + result->second);
            		}
        	}
        	gram_query.push_back(line);
        	_map.clear();
    	}

    	u64 query_start = getTime();

    	for (i = 0; i < gram_query.size(); ++i)
	{
		query q(table, i);

        	int min_bound,max_bound;
        	min_bound = (int)gram_query[i].size()*(1 - config.edit_distance_diff) - 1;
        	max_bound = (int)gram_query[i].size()*(1 + config.edit_distance_diff); // exclusive
    
        	if(min_bound < 0) min_bound = 0;
	    	if(max_bound > table.m_size()) max_bound = table.m_size();
        	for (j = 0; j < gram_query[i].size() ; ++j)
		{
			value = gram_query[i][j];
			if (value < 0)
			{
				continue;
			}

            for(int k = min_bound; k < max_bound; ++k)
			    q.attr(k, value, value, GPUGENIE_DEFAULT_WEIGHT, 0);
		}

		q.topk(config.num_of_topk);
		if (config.use_load_balance)
		{
			q.use_load_balance = true;
		}

		queries.push_back(q);
	}
    	u64 query_end = getTime();

    	cout<<"query build time = "<<getInterval(query_start, query_end)<<"ms."<<endl;
	u64 endtime = getTime();
	double timeInterval = getInterval(starttime, endtime);
	Logger::log(Logger::INFO, "%d queries are created!", queries.size());
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: loading query takes %f ms<<<<",
			timeInterval);
}




void GPUGenie::load_table_bijectMap(inv_table& table,
		std::vector<std::vector<int> >& data_points, GPUGenie_Config& config)
{
	u64 starttime = getTime();

	inv_list list;
    if(config.use_subsequence_search)
        list.invert_subsequence(data_points);
    else
	    list.invert_bijectMap(data_points);
	table.append(list);
	table.build(config.posting_list_max_length, config.use_load_balance);

	if (config.save_to_gpu)
		table.cpy_data_to_gpu();
	table.is_stored_in_gpu = config.save_to_gpu;

	u64 endtime = getTime();
	double timeInterval = getInterval(starttime, endtime);
	Logger::log(Logger::DEBUG,
			"Before finishing loading. i_size():%d, m_size():%d.",
			table.i_size(), table.m_size());
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: loading index takes %f ms (for one dim multi-values)<<<<",
			timeInterval);

}

void GPUGenie::load_table_bijectMap(inv_table& table, int *data,
		unsigned int item_num, unsigned int *index, unsigned int row_num,
		GPUGenie_Config& config)
{

	u64 starttime = getTime();

	inv_list list;
    if(config.use_subsequence_search)
        list.invert_subsequence(data, item_num, index, row_num);
    else
	    list.invert_bijectMap(data, item_num, index, row_num);

	table.append(list);
	table.build(config.posting_list_max_length, config.use_load_balance);

	if (config.save_to_gpu)
		table.cpy_data_to_gpu();
	table.is_stored_in_gpu = config.save_to_gpu;

	u64 endtime = getTime();
	double timeInterval = getInterval(starttime, endtime);
	Logger::log(Logger::DEBUG,
			"Before finishing loading. i_size():%d, m_size():%d.",
			table.i_size(), table.m_size());
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: loading index takes %f ms (for one dim multi-values)<<<<",
			timeInterval);

}

void GPUGenie::load_table_sequence(inv_table& table, vector<vector<int> >& data_points, GPUGenie_Config& config)
{
    u64 starttime = getTime();
    int min_value, max_value;
    vector<vector<int> > converted_data;
    vector<vector<int> > gram_data;
    sequence_reduce_to_ground(data_points, converted_data ,min_value ,max_value);
    table.set_min_value_sequence(min_value);
    table.set_max_value_sequence(max_value);
    sequence_to_gram(converted_data, gram_data, max_value-min_value ,config.data_gram_length);
    table.set_gram_length_sequence(config.data_gram_length);
    
    vector<vector<int> > length_id;//th 1st element, length is 1
    vector<inv_list> lists;
    for(unsigned int i = 0; i < gram_data.size(); ++i)
    {
        if(gram_data[i].size() <= 0)
            continue;
        if(length_id.size() < gram_data[i].size())
            length_id.resize(gram_data[i].size());
        length_id[gram_data[i].size()-1].push_back(i);

    }
    
    lists.resize(length_id.size());
    config.dim = length_id.size();
    cout<<"Start building index"<<endl;
    u64 tt1 = getTime();
    for(unsigned int i = 0; i < length_id.size(); ++i)
    {
        vector<vector<int> > temp_set;
        vector<int> respective_id;
        respective_id = length_id[i];
        for(unsigned int j = 0; j < length_id[i].size(); ++j)
            temp_set.push_back(gram_data[length_id[i][j]]);

        inv_list list;
        list.invert_sequence(temp_set, table.shift_bits_sequence, respective_id);
        table.append_sequence(list);
    }

    table.build(config.posting_list_max_length, config.use_load_balance);
    u64 tt2 = getTime();

    cout<<"Building table time = "<<getInterval(tt1, tt2)<<endl;

    if(config.save_to_gpu)
        table.cpy_data_to_gpu();
    table.is_stored_in_gpu = config.save_to_gpu;

    u64 endtime = getTime();
	double timeInterval = getInterval(starttime, endtime);
	Logger::log(Logger::DEBUG,
			"Before finishing loading. i_size():%d, m_size():%d.",
			table.i_size(), table.m_size());
	Logger::log(Logger::VERBOSE,
			">>>>[time profiling]: loading index takes %f ms (for one dim multi-values)<<<<",
			timeInterval);


    
}

void GPUGenie::knn_search_for_binary_data(std::vector<int>& result,
		std::vector<int>& result_count, GPUGenie_Config& config)
{
	
	inv_table *_table = NULL;

	preprocess_for_knn_binary(config, _table);

	knn_search_after_preprocess(config, _table, result, result_count);

	delete[] _table;
}

void GPUGenie::knn_search_for_csv_data(std::vector<int>& result,
		std::vector<int>& result_count, GPUGenie_Config& config)
{
	inv_table *_table = NULL;

	Logger::log(Logger::VERBOSE, "Starting preprocessing!");
	preprocess_for_knn_csv(config, _table);

	Logger::log(Logger::VERBOSE, "preprocessing finished!");

	knn_search_after_preprocess(config, _table, result, result_count);

	delete[] _table;
}

void GPUGenie::knn_search(std::vector<int>& result, GPUGenie_Config& config)
{
	std::vector<int> result_count;
	knn_search(result, result_count, config);
}

void GPUGenie::knn_search(std::vector<int>& result,
		std::vector<int>& result_count, GPUGenie_Config& config)
{
	try{
		u64 starttime = getTime();
		switch (config.data_type)
		{
		case 0:
			Logger::log(Logger::INFO, "search for csv data!");
			knn_search_for_csv_data(result, result_count, config);
			cout<<"knn for csv finished!"<<endl;
            break;
		case 1:
			Logger::log(Logger::INFO, "search for binary data!");
			knn_search_for_binary_data(result, result_count, config);
			break;
		default:
			throw GPUGenie::cpu_runtime_error("Please check data type in config\n");
		}

		u64 endtime = getTime();
		double elapsed = getInterval(starttime, endtime);

		Logger::log(Logger::VERBOSE,
				">>>>[time profiling]: knn_search totally takes %f ms (building query+match+selection)<<<<",
				elapsed);
	}
	catch (thrust::system::system_error &e){
        cout<<"system_error : "<<e.what()<<endl;
		throw GPUGenie::gpu_runtime_error(e.what());
	} catch (GPUGenie::gpu_bad_alloc &e){
        cout<<"bad_alloc"<<endl;
		throw e;
	} catch (GPUGenie::gpu_runtime_error &e){
		cout<<"run time error"<<endl;
        throw e;
	} catch(std::bad_alloc &e){
        cout<<"cpu bad alloc"<<endl;
		throw GPUGenie::cpu_bad_alloc(e.what());
	} catch(std::exception &e){
        cout<<"cpu runtime"<<endl;
		throw GPUGenie::cpu_runtime_error(e.what());
	} catch(...){
        cout<<"other error"<<endl;
		std::string msg = "Warning: Unknown Exception! Please try again or contact the author.\n";
		throw GPUGenie::cpu_runtime_error(msg.c_str());
	}
}

void GPUGenie::knn_search(inv_table& table, std::vector<query>& queries,
		std::vector<int>& h_topk, std::vector<int>& h_topk_count,
		GPUGenie_Config& config)
{
	int device_count, hashtable_size;
	cudaCheckErrors(cudaGetDeviceCount(&device_count));
	if (device_count == 0)
	{
		throw GPUGenie::cpu_runtime_error("NVIDIA CUDA-SUPPORTED GPU NOT FOUND! Program aborted..");
	}
	else if (device_count <= config.use_device)
	{
		Logger::log(Logger::INFO,
				"[Info] Device %d not found! Changing to %d...",
				config.use_device, GPUGENIE_DEFAULT_DEVICE);
		config.use_device = GPUGENIE_DEFAULT_DEVICE;
	}
	cudaCheckErrors(cudaSetDevice(config.use_device));

	Logger::log(Logger::INFO, "Using device %d...", config.use_device);
	Logger::log(Logger::DEBUG, "table.i_size():%d, config.hashtable_size:%f.",
			table.i_size(), config.hashtable_size);

	if (config.hashtable_size <= 2)
	{
		hashtable_size = table.i_size() * config.hashtable_size + 1;
	}
	else
	{
		hashtable_size = config.hashtable_size;
	}
	thrust::device_vector<int> d_topk, d_topk_count;

	int max_load = config.multiplier * config.posting_list_max_length + 1;

	Logger::log(Logger::DEBUG, "max_load is %d", max_load);

	GPUGenie::knn_bijectMap(
			table, //basic API, since encode dimension and value is also finally transformed as a bijection map
			queries, d_topk, d_topk_count, hashtable_size, max_load,
			config.count_threshold);

	Logger::log(Logger::INFO, "knn search is done!");
	Logger::log(Logger::DEBUG, "Topk obtained: %d in total.", d_topk.size());

	h_topk.resize(d_topk.size());
	h_topk_count.resize(d_topk_count.size());

	thrust::copy(d_topk.begin(), d_topk.end(), h_topk.begin());
	thrust::copy(d_topk_count.begin(), d_topk_count.end(),
			h_topk_count.begin());
}


void GPUGenie::reset_device()
{
    cudaDeviceReset();
}

void GPUGenie::get_rowID_offset(vector<int> &result, vector<int> &resultID,
                    vector<int> &resultOffset, unsigned int shift_bits)
{
    for(unsigned int i = 0 ; i < result.size() ; ++i)
    {
        int rowID, offset;
        rowID = result[i]>>shift_bits;
        offset = result[i] - (rowID<<shift_bits);
        resultID.push_back(rowID);
        resultOffset.push_back(offset);
    }
}

void GPUGenie::sequence_to_gram(vector<vector<int> > & sequences, vector<vector<int> >& gram_data,
        int max_value, int gram_length)
{
    int num_of_value = max_value + 1;
    int gram_value = 0;

    for(unsigned int i = 0 ; i < sequences.size() ; ++i)
    {
        int max_index = sequences[i].size() - gram_length + 1;
        if(max_index <= 0)
        {
            vector<int> line_null;
            vector<int> temp_line;
            for(unsigned int p = 0; p < (unsigned int)gram_length; ++p)
            {
                if(p<sequences[i].size() <= 0) break;
                if(p < sequences[i].size())
                    line_null.push_back(sequences[i][p]);
                else
                    line_null.push_back(sequences[i][0]);

            }
            int number = 0;
            for(unsigned int p = 0; p < (unsigned int)gram_length; ++p)
            {
                 number = number*num_of_value + line_null[p];
            }
            temp_line.push_back(number);
            gram_data.push_back(temp_line);
            continue;
        }
        vector<int> line;
        for(unsigned int j = 0 ; j < (unsigned int)max_index ; ++j)
        {
            gram_value = 0;
            for(unsigned int k = 0 ; k < (unsigned int)gram_length ; ++k )
            {
                gram_value = gram_value*num_of_value + sequences[i][j + k];
            }
            line.push_back(gram_value);
        }
        gram_data.push_back(line);
    }
}
void GPUGenie::sequence_reduce_to_ground(vector<vector<int> > & data, vector<vector<int> > & converted_data ,int& min_value ,int &max_value)
{
    min_value = data[0][0];
    max_value = min_value;
    converted_data.clear();
    for(unsigned int i = 0 ; i < data.size() ; ++i)
        for(unsigned int j = 0 ; j < data[i].size() ; ++j)
        {
            if(data[i][j] > max_value) max_value = data[i][j];
            if(data[i][j] < min_value) min_value = data[i][j];
        }
    for(unsigned int i = 0 ; i < data.size() ; ++i)
    {
        vector<int> line;
        for(unsigned int j = 0 ; j < data[i].size() ; ++j)
            line.push_back(data[i][j] - min_value);
        converted_data.push_back(line);
    
    }
}

