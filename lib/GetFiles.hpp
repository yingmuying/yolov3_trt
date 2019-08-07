#ifndef GETFILES_HPP_
#define GETFILES_HPP_

#include<vector>
#include<string>

#ifdef _MSC_VER    //windows下
#include<io.h>
#else
#include<dirent.h>    //linux下
#include<cstring>
#endif

std::string getName(const std::string& str){
#ifdef _MSC_VER    //windows 下的实现
	std::string name = str.substr(str.find_last_of("\\") + 1);    //   ****\\test.jpg   -->   test.jpg
	name = name.substr(0, name.find_first_of("."));          //   test.jpg   -->>   test
#else
	string name = str.substr(str.find_last_of("/") + 1);
	name = name.substr(0, name.find_first_of("."));
#endif
	return name;
}

void getFilesName(std::string path, std::vector<std::string> &files, bool recursion = true){
	using std::string;
	using std::vector;
#ifdef _MSC_VER    //windows 下的实现
	//文件句柄
	long long   hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1){
		do{
			//如果是目录
			if ((fileinfo.attrib &  _A_SUBDIR)){
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0){
					if(recursion)  //如果需要递归
						getFilesName(p.assign(path).append("\\").append(fileinfo.name), files);
					else   //不递归，直接保存目录名
						files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}
			}
			else //如果是文件
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	for (vector<string>::size_type i = 0; i < files.size(); i++){
		string str = files[i];
		int len = str.size();
		for (int k = 1; k < len - 1; k++)
			if (str[k] == '\\'&&str[k + 1] != '\\'&&str[k - 1] != '\\')
				str.insert(k, "\\");
		files[i] = str;
	}
}
#else
	std::string dirname = path;
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dirname.c_str())) == NULL)
		return;

	while ((dirp = readdir(dp)) != NULL){
		if ((strcmp(dirp->d_name, ".") == 0) || (strcmp(dirp->d_name, "..") == 0))
			continue;
		std::string temp = dirname;
		temp += "/";
		temp += dirp->d_name;
		if (dirp->d_type == 4)    //如果是目录
			if (recursion)  //如果需要递归
				getFilesName(temp, files);
			else
				files.push_back(temp);
		else if (dirp->d_type == 8)   //如果是文件
			files.push_back(temp);
		else
			;
	}
	closedir(dp);
}
#endif

#endif