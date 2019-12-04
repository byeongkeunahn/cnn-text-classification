
#pragma once


template<typename T>
class Vector
{
public:
    Vector() {

    }
    ~Vector() {
        
    }

public:
    void push_back(T& value);
    size_t size();

private:
    T *m_ptr;
};

