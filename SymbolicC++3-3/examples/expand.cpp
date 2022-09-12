/*
    SymbolicC++ : An object oriented computer algebra system written in C++

    Copyright (C) 2008 Yorick Hardy and Willi-Hans Steeb

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/


// expand.cpp

#include <iostream>
#include "symbolicc++.h"
using namespace std;

int main(void)
{
 Symbolic a("a"), b("b"), c("c"), y, z;

 y = (Symbolic(7)^3); cout << " y = " << y << endl;
 y = (a^0);           cout << " y = " << y << endl;
 y = (a^3);           cout << " y = " << y << endl;
 cout << endl;

 y = ((a+b-c)^3);     cout << " y = " << y << endl;
 cout << endl;

 y = (a+b)*(a-c);     cout << " y = " << y << endl;
 cout << endl;

 y = a+b;
 z = (y^4);           cout << " z = " << z << endl;
 return 0;
}
