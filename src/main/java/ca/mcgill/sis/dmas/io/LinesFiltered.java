/*******************************************************************************
 * Copyright 2015 McGill University. All rights reserved.                       
 *                                                                               
 * Unless required by applicable law or agreed to in writing, the software      
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF      
 * ANY KIND, either express or implied.                                         
 *******************************************************************************/
package ca.mcgill.sis.dmas.io;

import java.util.Iterator;
import java.util.regex.Pattern;

import ca.mcgill.sis.dmas.env.StringResources;

public class LinesFiltered extends Lines{
	
	String regex = StringResources.STR_EMPTY;
	Lines lines;
	boolean skipEmpty = false;
	public LinesFiltered(Lines lines, String regexPattern, boolean skipEmpty){
		regex = regexPattern;
		this.lines = lines;
		this.skipEmpty = skipEmpty;
	}

	@Override
	public Iterator<String> iterator() {
		return new LinesFilteredIterator();
	}
	
	private class LinesFilteredIterator implements Iterator<String>{
		
		Pattern pattern = Pattern.compile(regex);

		Iterator<String> ite = lines.iterator();
		String thisline = null;
		
		@Override
		public boolean hasNext() {
			while (thisline == null || pattern.matcher(thisline).find() || (skipEmpty && thisline.trim().length() == 0)) {
				if(ite.hasNext())
					thisline = ite.next();
				else
					break;
			}
			if(thisline == null || pattern.matcher(thisline).find())
				return false;
			else
				return true;
		}

		@Override
		public String next() {
			String returnLine = thisline;
			thisline = null;
			return returnLine;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
		
	}
	
	public static void main(String [] args){
	}

}
