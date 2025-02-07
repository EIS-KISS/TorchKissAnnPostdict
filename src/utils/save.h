/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <valarray>

#include "log.h"

namespace csv
{
	template<typename DataType>
	void save2dTensorToCsv(const torch::Tensor& tensor, std::iostream& stream);

	template<typename DataType>
	void save1dTensorToCsv(const torch::Tensor& tensor, std::iostream& stream);

	bool save(const std::filesystem::path& path, const torch::Tensor& tensor, const std::string& name);

	template <typename T>
	bool save(const std::filesystem::path& path, const std::vector<T>& data, const std::string& name);

	template <typename T>
	bool save(const std::filesystem::path& path, const std::valarray<T>& data, const std::string& name);

};

template<typename DataType>
void csv::save2dTensorToCsv(const torch::Tensor& tensor, std::iostream& stream)
{
	auto tensorAcesssor = tensor.accessor<DataType, 2>();
	for(int i = 0; i < tensorAcesssor.size(0); ++i)
	{
		for(int j = 0; j < tensorAcesssor.size(1); ++j)
		{
			stream<<tensorAcesssor[i][j];
			if(j+1 < tensorAcesssor.size(1))
				stream<<", ";
		}
		stream<<'\n';
	}
}

template<typename DataType>
void csv::save1dTensorToCsv(const torch::Tensor& tensor, std::iostream& stream)
{
	auto tensorAcesssor = tensor.accessor<DataType, 1>();

	for(int i = 0; i < tensorAcesssor.size(0); ++i)
	{
		stream<<tensorAcesssor[i];
		if(i+1 < tensorAcesssor.size(0))
			stream<<", ";
	}
	stream<<'\n';
}

template <typename T>
bool csv::save(const std::filesystem::path& path, const std::vector<T>& data, const std::string& name)
{
	return csv::save(path, std::valarray<T>(data.data(), data.size()), name);
}

template <typename T>
bool csv::save(const std::filesystem::path& path, const std::valarray<T>& data, const std::string& name)
{
	std::fstream file;
	file.open(path, std::ios_base::out);
	if(!file.is_open())
	{
		Log(Log::ERROR)<<"can not open "<<path<<" for writing\n";
		return false;
	}

	file<<std::scientific;
	file<<name<<'\n';

	for(size_t i = 0; i < data.size(); ++i)
	{
		file<<data[i];
		if(i+1 < data.size())
			file<<", ";
	}
	file<<'\n';

	file.close();
	return true;
}
