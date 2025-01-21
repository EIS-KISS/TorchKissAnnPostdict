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

#pragma once
#include <filesystem>
#include <kisstype/spectra.h>
#include <eisgenerator/translators.h>
#include <memory>

#include "data/loaders/eisspectradataset.h"

class DirDataset: public EisSpectraDataset
{
protected:
	struct File
	{
		std::filesystem::path path;
	};

	std::shared_ptr<std::vector<File>> files;

	void loadDir(const std::filesystem::path& path);
	virtual eis::Spectra loadSpectraAtIndex(size_t index) override;
	virtual eis::Spectra loadSpectraHeaderAtIndex(size_t index) override;

public:
	DirDataset() = default;
};
