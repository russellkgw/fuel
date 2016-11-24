class DataImportController < ApplicationController

  def index
    @headers = { Data::Handler::EXCHANGE_RATE_FILE => Data::ExchangeRateImport::VALID_HEADERS,
                 Data::Handler::OIL_PRICE_FILE => Data::ExchangeRateImport::VALID_HEADERS,
                 Data::Handler::FUEL_PRICE_FILE => Data::ExchangeRateImport::VALID_HEADERS }
  end

  def create
    flash[:notice] = Data::Handler.process_csv_files(data_files)
    redirect_to action: 'index'
  end

  private

  def data_files
    { Data::Handler::EXCHANGE_RATE_FILE => params[Data::Handler::EXCHANGE_RATE_FILE].try(:tempfile),
      Data::Handler::OIL_PRICE_FILE => params[Data::Handler::OIL_PRICE_FILE].try(:tempfile),
      Data::Handler::FUEL_PRICE_FILE => params[Data::Handler::FUEL_PRICE_FILE].try(:tempfile) }
  end

end